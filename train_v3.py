import json
import gc
import ast
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# ============================================================
# V3: configuration
# ============================================================
TRAIN_FILE = "train.csv"
MODEL_FILE = "ctr_model_v3.txt"
CHUNK_SIZE = 200_000

TRAIN_END_DATE = "2025-01-24"
VALID_START_DATE = "2025-01-25"
VALID_END_DATE = "2025-01-26"

SMOOTHING = 50.0
PRIOR_CTR = 0.06

FEATURE_NAMES = [
    # V1 features
    "item_ctr",
    "user_ctr",
    "repeat_click",
    "history_count",
    "history_length",
    "history_unique_count",
    "history_duplicate_ratio",
    "time_bucket",
    "day_type",
    # V3 temporal history features
    "history_recent_1h",
    "history_recent_6h",
    "history_recent_24h",
    "history_recent_72h",
    "history_last_age_hours",
    "history_mean_age_hours",
    "history_span_hours",
    "history_item_last_age_hours",
    "history_item_recent_24h",
]

USE_COLUMNS = [
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
        # Defensive fallback for slightly non-standard serialized lists.
        try:
            obj = ast.literal_eval(value)
            return obj if isinstance(obj, list) else []
        except (ValueError, SyntaxError, TypeError):
            return []


def parse_history_pair(item_value, timestamp_value, row_timestamp, item_id):
    """Return compact, leakage-safe temporal features from this row's history only."""
    items = parse_json_list(item_value)
    times = parse_json_list(timestamp_value)

    # If malformed/misaligned, keep item-history features but don't fabricate timestamps.
    if not isinstance(items, list):
        items = []
    if not isinstance(times, list):
        times = []

    history_length = len(items)
    history_unique_count = len(set(items))
    history_count = sum(1 for x in items if x == item_id)
    duplicate_ratio = (
        (history_length - history_unique_count) / history_length
        if history_length else 0.0
    )

    valid_ages = []
    item_ages = []
    if len(items) == len(times):
        for hist_item, hist_ts in zip(items, times):
            try:
                age_seconds = float(row_timestamp) - float(hist_ts)
            except (ValueError, TypeError):
                continue
            # Only accept events strictly before the current row.
            if age_seconds < 0:
                continue
            age_hours = age_seconds / 3600.0
            valid_ages.append(age_hours)
            if hist_item == item_id:
                item_ages.append(age_hours)

    if valid_ages:
        last_age = min(valid_ages)
        mean_age = float(np.mean(valid_ages))
        span_hours = max(valid_ages) - min(valid_ages)
        recent_1h = sum(a <= 1 for a in valid_ages)
        recent_6h = sum(a <= 6 for a in valid_ages)
        recent_24h = sum(a <= 24 for a in valid_ages)
        recent_72h = sum(a <= 72 for a in valid_ages)
    else:
        last_age = mean_age = span_hours = 0.0
        recent_1h = recent_6h = recent_24h = recent_72h = 0

    item_last_age = min(item_ages) if item_ages else 0.0
    item_recent_24h = sum(a <= 24 for a in item_ages)

    # Add explicit presence flags so zero age isn't confused with missing history.
    return [
        history_count,
        history_length,
        history_unique_count,
        duplicate_ratio,
        int(history_length > 0),
        recent_1h,
        recent_6h,
        recent_24h,
        recent_72h,
        min(last_age, 24 * 365.0),
        min(mean_age, 24 * 365.0),
        min(span_hours, 24 * 365.0),
        min(item_last_age, 24 * 365.0),
        item_recent_24h,
    ]


def iter_daily_frames():
    """Requires train.csv ordered chronologically; emits complete UTC days."""
    pending = None
    reader = pd.read_csv(TRAIN_FILE, usecols=USE_COLUMNS, chunksize=CHUNK_SIZE)
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
        for key, show_dict, click_dict in [
            ("item_id", self.item_show, self.item_click),
            ("user_id", self.user_show, self.user_click),
        ]:
            agg = df.groupby(key)["clicked"].agg(["size", "sum"])
            for entity, row in agg.iterrows():
                show_dict[entity] += int(row["size"])
                click_dict[entity] += int(row["sum"])
        self.total_rows += len(df)
        self.total_clicks += int(df["clicked"].sum())

    def copy(self):
        import copy
        return copy.deepcopy(self)


def build_features(df, stats):
    out = np.empty((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    for pos, row in enumerate(df.itertuples(index=False)):
        item = row.item_id
        user = row.user_id
        ts = row.timestamp

        hist_features = parse_history_pair(
            row.history_item_ids,
            row.history_timestamps,
            ts,
            item,
        )

        history_count = hist_features[0]
        history_length = hist_features[1]
        unique_count = hist_features[2]
        duplicate_ratio = hist_features[3]
        history_present = hist_features[4]
        recent_1h, recent_6h, recent_24h, recent_72h = hist_features[5:9]
        last_age, mean_age, span_hours, item_last_age, item_recent_24h = hist_features[9:14]

        # Preserve the original nine features, then add temporal history features.
        out[pos] = [
            stats.item_ctr(item),
            stats.user_ctr(user),
            int(history_count > 0),
            history_count,
            history_length,
            unique_count,
            duplicate_ratio,
            row.time_bucket,
            row.day_type,
            recent_1h,
            recent_6h,
            recent_24h,
            recent_72h,
            last_age,
            mean_age,
            span_hours,
            item_last_age,
            item_recent_24h,
        ]
    return out


print("=" * 60)
print("V3: history-timestamp feature engineering")
print("=" * 60)

stats = HistoricalStats()
daily_features = {}
stats_at_train_end = None

for date, day_df in iter_daily_frames():
    print(f"Processing {date}: {len(day_df):,} rows")
    # The history fields are row-local; CTRs use prior dates only.
    X_day = build_features(day_df, stats)
    y_day = day_df["clicked"].to_numpy(dtype=np.int8)
    daily_features[date] = (X_day, y_day)
    stats.update(day_df)
    if date == TRAIN_END_DATE:
        stats_at_train_end = stats.copy()
    del day_df
    gc.collect()

if stats_at_train_end is None:
    raise RuntimeError(f"Could not find training cutoff date {TRAIN_END_DATE}.")

all_dates = sorted(daily_features)
train_dates = [d for d in all_dates if d <= TRAIN_END_DATE]
valid_dates = [d for d in all_dates if VALID_START_DATE <= d <= VALID_END_DATE]
if not train_dates or not valid_dates:
    raise RuntimeError("Empty training or validation split; check date ranges.")

print("\nTraining dates:", train_dates)
print("Validation dates:", valid_dates)

X_train = np.concatenate([daily_features[d][0] for d in train_dates], axis=0)
y_train = np.concatenate([daily_features[d][1] for d in train_dates], axis=0)

print("\nBuilding validation features with frozen Jan 24 CTR stats")
X_valid_parts, y_valid_parts = [], []
for date, day_df in iter_daily_frames():
    if date not in valid_dates:
        del day_df
        continue
    print(f"Validation features for {date}: {len(day_df):,} rows")
    X_valid_parts.append(build_features(day_df, stats_at_train_end))
    y_valid_parts.append(day_df["clicked"].to_numpy(dtype=np.int8))
    del day_df
    gc.collect()

X_valid = np.concatenate(X_valid_parts, axis=0)
y_valid = np.concatenate(y_valid_parts, axis=0)

print("\nX_train:", X_train.shape, "X_valid:", X_valid.shape)
print("Feature count:", len(FEATURE_NAMES))
print("Training row CTR:", float(y_train.mean()))
print("Validation row CTR:", float(y_valid.mean()))

params = dict(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

print("\nTraining LightGBM V3")
model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=50),
    ],
)

valid_pred = model.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, valid_pred)
best_iteration = model.best_iteration_ or params["n_estimators"]

print("\nV3 validation AUC:", valid_auc)
print("Best iteration:", best_iteration)

X_all = np.concatenate([daily_features[d][0] for d in all_dates], axis=0)
y_all = np.concatenate([daily_features[d][1] for d in all_dates], axis=0)
del X_train, X_valid, y_train, y_valid, X_valid_parts, y_valid_parts
del valid_pred, model
gc.collect()

final_params = params.copy()
final_params["n_estimators"] = best_iteration
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X_all, y_all)
final_model.booster_.save_model(MODEL_FILE)

print("\nSaved model:", MODEL_FILE)
print("Feature count:", len(FEATURE_NAMES))
print("Final training rows:", len(y_all))
print("V3 validation AUC:", valid_auc)
print("Done.")
