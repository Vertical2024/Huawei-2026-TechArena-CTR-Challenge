import json
import gc
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# ============================================================
# Configuration
# ============================================================
TRAIN_FILE = "train.csv"
MODEL_FILE = "ctr_model_v2.txt"

CHUNK_SIZE = 200_000
TRAIN_END_DATE = "2025-01-24"
VALID_START_DATE = "2025-01-25"
VALID_END_DATE = "2025-01-26"

SMOOTHING = 50.0
PRIOR_CTR = 0.06

# A daily contribution is multiplied by DECAY_FACTOR for each day
# between its observation date and the date being predicted.
# 0.70 gives recent days more weight without discarding older history.
DECAY_FACTOR = 0.70
ROLLING_WINDOWS = (1, 3, 7)

FEATURE_NAMES = [
    "item_ctr",
    "user_ctr",
    "item_ctr_1d",
    "item_ctr_3d",
    "item_ctr_7d",
    "item_ctr_decay",
    "user_ctr_decay",
    "repeat_click",
    "history_count",
    "history_length",
    "history_unique_count",
    "history_duplicate_ratio",
    "time_bucket",
    "day_type",
]

USE_COLUMNS = [
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
    """Input CSV must be chronologically sorted; emit complete UTC dates."""
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


class TemporalStats:
    """Cumulative, rolling-window, and exponentially decayed item/user CTR stats."""

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

        # Keep daily aggregates for exact 1/3/7-day historical windows.
        self.daily_item = deque(maxlen=max(ROLLING_WINDOWS))
        self.daily_user = deque(maxlen=max(ROLLING_WINDOWS))
        self.daily_global = deque(maxlen=max(ROLLING_WINDOWS))

    def global_ctr(self):
        if self.global_rows <= 0:
            return PRIOR_CTR
        return self.global_clicks / self.global_rows

    def _smoothed_ctr(self, clicks, shows, prior):
        return (clicks + SMOOTHING * prior) / (shows + SMOOTHING)

    def item_ctr(self, item):
        prior = self.global_ctr()
        return self._smoothed_ctr(self.item_click[item], self.item_show[item], prior)

    def user_ctr(self, user):
        prior = self.global_ctr()
        return self._smoothed_ctr(self.user_click[user], self.user_show[user], prior)

    def _window_ctr(self, key, index, window, prior):
        # index 0 = item, index 1 = user
        daily = self.daily_item if index == 0 else self.daily_user
        show_key = "show"
        click_key = "click"
        show = 0.0
        click = 0.0
        for day_map in list(daily)[-window:]:
            show += day_map[show_key].get(key, 0)
            click += day_map[click_key].get(key, 0)
        return self._smoothed_ctr(click, show, prior)

    def item_ctr_window(self, item, window):
        return self._window_ctr(item, 0, window, self.global_ctr())

    def item_ctr_decay_value(self, item):
        prior = self.decay_global_clicks / self.decay_global_rows if self.decay_global_rows else PRIOR_CTR
        return self._smoothed_ctr(
            self.decay_item_click[item], self.decay_item_show[item], prior
        )

    def user_ctr_decay_value(self, user):
        prior = self.decay_global_clicks / self.decay_global_rows if self.decay_global_rows else PRIOR_CTR
        return self._smoothed_ctr(
            self.decay_user_click[user], self.decay_user_show[user], prior
        )

    @staticmethod
    def _aggregate(df, key):
        agg = df.groupby(key)["clicked"].agg(["size", "sum"])
        shows = {k: float(v) for k, v in agg["size"].items()}
        clicks = {k: float(v) for k, v in agg["sum"].items()}
        return {"show": shows, "click": clicks}

    def update(self, df):
        # Cumulative statistics
        for key, show_key, click_key in [
            ("item_id", self.item_show, self.item_click),
            ("user_id", self.user_show, self.user_click),
        ]:
            agg = df.groupby(key)["clicked"].agg(["size", "sum"])
            for entity, row in agg.iterrows():
                show_key[entity] += float(row["size"])
                click_key[entity] += float(row["sum"])

        # Decay all previously observed statistics by one day, then add today's data.
        for values in (
            self.decay_item_show, self.decay_item_click,
            self.decay_user_show, self.decay_user_click,
        ):
            for key in list(values.keys()):
                values[key] *= DECAY_FACTOR

        self.decay_global_rows *= DECAY_FACTOR
        self.decay_global_clicks *= DECAY_FACTOR

        item_agg = df.groupby("item_id")["clicked"].agg(["size", "sum"])
        for entity, row in item_agg.iterrows():
            self.decay_item_show[entity] += float(row["size"])
            self.decay_item_click[entity] += float(row["sum"])

        user_agg = df.groupby("user_id")["clicked"].agg(["size", "sum"])
        for entity, row in user_agg.iterrows():
            self.decay_user_show[entity] += float(row["size"])
            self.decay_user_click[entity] += float(row["sum"])

        self.global_rows += len(df)
        self.global_clicks += int(df["clicked"].sum())
        self.decay_global_rows += len(df)
        self.decay_global_clicks += float(df["clicked"].sum())

        self.daily_item.append(self._aggregate(df, "item_id"))
        self.daily_user.append(self._aggregate(df, "user_id"))
        self.daily_global.append({
            "show": float(len(df)),
            "click": float(df["clicked"].sum()),
        })

    def copy(self):
        # Used only at the validation cutoff; deep-copy dictionaries and daily aggregates.
        import copy
        return copy.deepcopy(self)


def build_features(df, stats):
    features = []
    for row in df.itertuples(index=False):
        history = parse_history(row.history_item_ids)
        item = row.item_id
        user = row.user_id
        history_count = history.count(item)
        history_length = len(history)
        history_unique_count = len(set(history))
        duplicate_ratio = (
            (history_length - history_unique_count) / history_length
            if history_length else 0.0
        )
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
            history_unique_count,
            duplicate_ratio,
            row.time_bucket,
            row.day_type,
        ])
    return np.asarray(features, dtype=np.float32)


print("=" * 60)
print("V2: Building chronological training features")
print("=" * 60)

stats = TemporalStats()
daily_features = {}
stats_at_train_end = None

for date, day_df in iter_daily_frames():
    print(f"Processing {date}: {len(day_df):,} rows")
    X_day = build_features(day_df, stats)  # features before current-day labels
    y_day = day_df["clicked"].to_numpy(dtype=np.int8)
    daily_features[date] = (X_day, y_day)
    stats.update(day_df)  # update only after features for the day are complete
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
    raise RuntimeError("Empty training or validation split; check dates.")

print("\nTraining dates:", train_dates)
print("Validation dates:", valid_dates)

X_train = np.concatenate([daily_features[d][0] for d in train_dates], axis=0)
y_train = np.concatenate([daily_features[d][1] for d in train_dates], axis=0)

print("\nBuilding validation features with frozen Jan 24 statistics")
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

print("\nTraining LightGBM V2")
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

print("\nV2 validation AUC:", valid_auc)
print("Best iteration:", best_iteration)

# Rebuild final features using all training dates' strictly prior history.
# The model trains on the chronological features already stored above.
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
print("V2 validation AUC:", valid_auc)
print("Done.")
