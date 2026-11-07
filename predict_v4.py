import json
import gc
import ast
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb


TRAIN_FILE = "train.csv"
EVAL_FILE = "evaluation.csv"
MODEL_FILE = "ctr_model_v4.txt"
OUTPUT_FILE = "submission_v4.csv"
CHUNK_SIZE = 200_000
SMOOTHING = 50.0
PRIOR_CTR = 0.06

TRAIN_COLUMNS = [
    "event_id", "user_id", "item_id", "timestamp",
    "history_item_ids", "history_timestamps",
    "time_bucket", "day_type", "clicked",
]


def parse_json_list(value):
    if not isinstance(value, str) or not value:
        return []
    try:
        obj = json.loads(value)
        return obj if isinstance(obj, list) else []
    except (json.JSONDecodeError, TypeError):
        try:
            obj = ast.literal_eval(value)
            return obj if isinstance(obj, list) else []
        except (ValueError, SyntaxError, TypeError):
            return []


def parse_history_pair(item_value, timestamp_value, row_timestamp, item_id):
    items = parse_json_list(item_value)
    times = parse_json_list(timestamp_value)
    history_length = len(items)
    unique_count = len(set(items))
    history_count = sum(1 for x in items if x == item_id)
    duplicate_ratio = (history_length - unique_count) / history_length if history_length else 0.0

    ages = []
    item_ages = []
    if len(items) == len(times):
        for hist_item, hist_ts in zip(items, times):
            try:
                age = float(row_timestamp) - float(hist_ts)
            except (ValueError, TypeError):
                continue
            if age < 0:
                continue
            age_h = age / 3600.0
            ages.append(age_h)
            if hist_item == item_id:
                item_ages.append(age_h)

    if ages:
        last_age = min(ages)
        mean_age = float(np.mean(ages))
        span = max(ages) - min(ages)
        recent_1h = sum(a <= 1 for a in ages)
        recent_6h = sum(a <= 6 for a in ages)
        recent_24h = sum(a <= 24 for a in ages)
        recent_72h = sum(a <= 72 for a in ages)
    else:
        last_age = mean_age = span = 0.0
        recent_1h = recent_6h = recent_24h = recent_72h = 0

    item_last_age = min(item_ages) if item_ages else 0.0
    item_recent_24h = sum(a <= 24 for a in item_ages)
    item_recent_6h = sum(a <= 6 for a in item_ages)
    item_recent_72h = sum(a <= 72 for a in item_ages)
    item_recency_score = float(sum(2.0 ** (-a / 24.0) for a in item_ages))

    return [
        history_count, history_length, unique_count, duplicate_ratio,
        int(history_length > 0), recent_1h, recent_6h, recent_24h, recent_72h,
        min(last_age, 24 * 365.0), min(mean_age, 24 * 365.0),
        min(span, 24 * 365.0), min(item_last_age, 24 * 365.0),
        item_recent_24h, item_recent_6h, item_recent_72h, item_recency_score,
    ]


class HistoricalStats:
    def __init__(self):
        self.item_show = defaultdict(int)
        self.item_click = defaultdict(int)
        self.user_show = defaultdict(int)
        self.user_click = defaultdict(int)
        self.total_rows = 0
        self.total_clicks = 0

    def global_ctr(self):
        return self.total_clicks / self.total_rows if self.total_rows else PRIOR_CTR

    def item_ctr(self, item):
        g = self.global_ctr()
        return (self.item_click[item] + SMOOTHING * g) / (self.item_show[item] + SMOOTHING)

    def user_ctr(self, user):
        g = self.global_ctr()
        return (self.user_click[user] + SMOOTHING * g) / (self.user_show[user] + SMOOTHING)

    def update(self, df):
        for key, shows, clicks in [
            ("item_id", self.item_show, self.item_click),
            ("user_id", self.user_show, self.user_click),
        ]:
            agg = df.groupby(key)["clicked"].agg(["size", "sum"])
            for entity, row in agg.iterrows():
                shows[entity] += int(row["size"])
                clicks[entity] += int(row["sum"])
        self.total_rows += len(df)
        self.total_clicks += int(df["clicked"].sum())


def build_features(df, stats):
    out = np.empty((len(df), 21), dtype=np.float32)
    for pos, row in enumerate(df.itertuples(index=False)):
        h = parse_history_pair(
            row.history_item_ids, row.history_timestamps, row.timestamp, row.item_id
        )
        out[pos] = [
            stats.item_ctr(row.item_id),
            stats.user_ctr(row.user_id),
            int(h[0] > 0),
            h[0], h[1], h[2], h[3],
            row.time_bucket, row.day_type,
            h[5], h[6], h[7], h[8],
            h[9], h[10], h[11], h[12], h[13], h[14], h[15], h[16],
        ]
    return out


def iter_daily_frames():
    pending = None
    reader = pd.read_csv(TRAIN_FILE, usecols=TRAIN_COLUMNS, chunksize=CHUNK_SIZE)
    for chunk in reader:
        chunk["date"] = pd.to_datetime(chunk["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        dates = chunk["date"].drop_duplicates().tolist()
        if len(dates) == 1:
            pending = chunk
            continue
        for date in dates[:-1]:
            yield date, chunk[chunk["date"] == date].copy()
        last_date = dates[-1]
        pending = chunk[chunk["date"] == last_date].copy()
    if pending is not None and len(pending):
        yield pending["date"].iloc[0], pending


print("Building historical CTR statistics from train.csv only...")
stats = HistoricalStats()
for date, day_df in iter_daily_frames():
    stats.update(day_df)
    print(f"Processed {date}: {len(day_df):,} rows")
    del day_df
    gc.collect()

model = lgb.Booster(model_file=MODEL_FILE)
print("Loaded model:", MODEL_FILE)
print("Model feature count:", model.num_feature())
if model.num_feature() != 21:
    raise ValueError(f"Expected a 21-feature V4 model, but {MODEL_FILE} has {model.num_feature()} features. Run train_v4.py first.")

rowids = []
predictions = []
print("Generating V4 predictions from each evaluation row's own history...")
for chunk in pd.read_csv(EVAL_FILE, chunksize=CHUNK_SIZE):
    if "rowid" not in chunk.columns:
        raise KeyError("evaluation.csv must contain a 'rowid' column.")
    X = build_features(chunk, stats)
    scores = model.predict(X)
    if not np.isfinite(scores).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    rowids.extend(chunk["rowid"].tolist())
    predictions.extend(scores.tolist())
    print("Predicted rows:", len(rowids))
    del X, chunk
    gc.collect()

submission = pd.DataFrame({
    "rowid": rowids,
    "predicted_probability": predictions,
})
submission.to_csv(OUTPUT_FILE, index=False)
print("\nSaved:", OUTPUT_FILE)
print("Rows:", len(submission))
print(submission["predicted_probability"].describe())
