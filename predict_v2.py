import json
import gc
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import lightgbm as lgb


# Must match train_v2.py
TRAIN_FILE = "train.csv"
EVAL_FILE = "evaluation.csv"
MODEL_FILE = "ctr_model_v2.txt"
OUTPUT_FILE = "submission_v2.csv"
CHUNK_SIZE = 200_000

SMOOTHING = 50.0
PRIOR_CTR = 0.06
DECAY_FACTOR = 0.70
ROLLING_WINDOWS = (1, 3, 7)

TRAIN_COLUMNS = [
    "event_id", "user_id", "item_id", "timestamp",
    "history_item_ids", "time_bucket", "day_type", "clicked",
]


def parse_history(value):
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def iter_daily_frames():
    pending = None
    reader = pd.read_csv(TRAIN_FILE, usecols=TRAIN_COLUMNS, chunksize=CHUNK_SIZE)
    for chunk in reader:
        chunk["date"] = pd.to_datetime(
            chunk["timestamp"], unit="s", utc=True
        ).dt.strftime("%Y-%m-%d")
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


class TemporalStats:
    def __init__(self):
        self.global_rows = 0
        self.global_clicks = 0
        self.item_show = defaultdict(float)
        self.item_click = defaultdict(float)
        self.user_show = defaultdict(float)
        self.user_click = defaultdict(float)
        self.decay_item_show = defaultdict(float)
        self.decay_item_click = defaultdict(float)
        self.decay_user_show = defaultdict(float)
        self.decay_user_click = defaultdict(float)
        self.decay_global_rows = 0.0
        self.decay_global_clicks = 0.0
        self.daily_item = deque(maxlen=max(ROLLING_WINDOWS))
        self.daily_user = deque(maxlen=max(ROLLING_WINDOWS))
        self.daily_global = deque(maxlen=max(ROLLING_WINDOWS))

    def global_ctr(self):
        return self.global_clicks / self.global_rows if self.global_rows else PRIOR_CTR

    def _smoothed_ctr(self, clicks, shows, prior):
        return (clicks + SMOOTHING * prior) / (shows + SMOOTHING)

    def item_ctr(self, item):
        return self._smoothed_ctr(self.item_click[item], self.item_show[item], self.global_ctr())

    def user_ctr(self, user):
        return self._smoothed_ctr(self.user_click[user], self.user_show[user], self.global_ctr())

    def _window_ctr(self, key, index, window, prior):
        daily = self.daily_item if index == 0 else self.daily_user
        show = click = 0.0
        for day_map in list(daily)[-window:]:
            show += day_map["show"].get(key, 0)
            click += day_map["click"].get(key, 0)
        return self._smoothed_ctr(click, show, prior)

    def item_ctr_window(self, item, window):
        return self._window_ctr(item, 0, window, self.global_ctr())

    def item_ctr_decay_value(self, item):
        prior = self.decay_global_clicks / self.decay_global_rows if self.decay_global_rows else PRIOR_CTR
        return self._smoothed_ctr(self.decay_item_click[item], self.decay_item_show[item], prior)

    def user_ctr_decay_value(self, user):
        prior = self.decay_global_clicks / self.decay_global_rows if self.decay_global_rows else PRIOR_CTR
        return self._smoothed_ctr(self.decay_user_click[user], self.decay_user_show[user], prior)

    @staticmethod
    def _aggregate(df, key):
        agg = df.groupby(key)["clicked"].agg(["size", "sum"])
        return {
            "show": {k: float(v) for k, v in agg["size"].items()},
            "click": {k: float(v) for k, v in agg["sum"].items()},
        }

    def update(self, df):
        for key, show_dict, click_dict in [
            ("item_id", self.item_show, self.item_click),
            ("user_id", self.user_show, self.user_click),
        ]:
            agg = df.groupby(key)["clicked"].agg(["size", "sum"])
            for entity, row in agg.iterrows():
                show_dict[entity] += float(row["size"])
                click_dict[entity] += float(row["sum"])

        for values in (
            self.decay_item_show, self.decay_item_click,
            self.decay_user_show, self.decay_user_click,
        ):
            for key in list(values.keys()):
                values[key] *= DECAY_FACTOR
        self.decay_global_rows *= DECAY_FACTOR
        self.decay_global_clicks *= DECAY_FACTOR

        for key, show_dict, click_dict in [
            ("item_id", self.decay_item_show, self.decay_item_click),
            ("user_id", self.decay_user_show, self.decay_user_click),
        ]:
            agg = df.groupby(key)["clicked"].agg(["size", "sum"])
            for entity, row in agg.iterrows():
                show_dict[entity] += float(row["size"])
                click_dict[entity] += float(row["sum"])

        self.global_rows += len(df)
        self.global_clicks += int(df["clicked"].sum())
        self.decay_global_rows += len(df)
        self.decay_global_clicks += float(df["clicked"].sum())
        self.daily_item.append(self._aggregate(df, "item_id"))
        self.daily_user.append(self._aggregate(df, "user_id"))
        self.daily_global.append({"show": float(len(df)), "click": float(df["clicked"].sum())})


def build_features(df, stats):
    features = []
    for row in df.itertuples(index=False):
        history = parse_history(row.history_item_ids)
        item, user = row.item_id, row.user_id
        history_count = history.count(item)
        history_length = len(history)
        unique_count = len(set(history))
        duplicate_ratio = (history_length - unique_count) / history_length if history_length else 0.0
        features.append([
            stats.item_ctr(item),
            stats.user_ctr(user),
            stats.item_ctr_window(item, 1),
            stats.item_ctr_window(item, 3),
            stats.item_ctr_window(item, 7),
            stats.item_ctr_decay_value(item),
            stats.user_ctr_decay_value(user),
            int(history_count > 0),
            history_count,
            history_length,
            unique_count,
            duplicate_ratio,
            row.time_bucket,
            row.day_type,
        ])
    return np.asarray(features, dtype=np.float32)


print("Building temporal statistics from training data...")
stats = TemporalStats()
for date, day_df in iter_daily_frames():
    stats.update(day_df)
    print(f"Processed {date}: {len(day_df):,} rows")
    del day_df
    gc.collect()

print("Global CTR:", stats.global_ctr())
model = lgb.Booster(model_file=MODEL_FILE)
print("Loaded model:", MODEL_FILE)

rowids = []
predictions = []
print("Generating V2 predictions...")

for chunk in pd.read_csv(EVAL_FILE, chunksize=CHUNK_SIZE):
    X = build_features(chunk, stats)
    scores = model.predict(X)
    if not np.isfinite(scores).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    if "rowid" not in chunk.columns:
        raise KeyError("evaluation.csv must contain a 'rowid' column.")
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
