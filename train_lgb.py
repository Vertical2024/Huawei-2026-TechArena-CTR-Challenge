import json
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = "train.csv"
MODEL_FILE = "ctr_model_improved.txt"

CHUNK_SIZE = 200_000

TRAIN_END_DATE = "2025-01-24"
VALID_START_DATE = "2025-01-25"
VALID_END_DATE = "2025-01-26"

SMOOTHING = 50.0

# Fixed prior for dates before the first available training day.
# Do not use the full-dataset label rate for this prior.
PRIOR_CTR = 0.06

FEATURE_NAMES = [
    "item_ctr",
    "user_ctr",
    "repeat_click",
    "history_count",
    "history_length",
    "history_unique_count",
    "history_duplicate_ratio",
    "time_bucket",
    "day_type",
]

USE_COLUMNS = [
    "event_id",
    "user_id",
    "item_id",
    "timestamp",
    "history_item_ids",
    "time_bucket",
    "day_type",
    "clicked",
]


# ============================================================
# Historical statistics
# ============================================================

class HistoricalStats:

    def __init__(self):

        self.item_show = defaultdict(int)
        self.item_click = defaultdict(int)

        self.user_show = defaultdict(int)
        self.user_click = defaultdict(int)

        self.total_rows = 0
        self.total_clicks = 0

    def global_ctr(self):

        if self.total_rows == 0:
            return PRIOR_CTR

        return self.total_clicks / self.total_rows

    def item_ctr(self, item):

        g = self.global_ctr()

        return (
            self.item_click[item] + SMOOTHING * g
        ) / (
            self.item_show[item] + SMOOTHING
        )

    def user_ctr(self, user):

        g = self.global_ctr()

        return (
            self.user_click[user] + SMOOTHING * g
        ) / (
            self.user_show[user] + SMOOTHING
        )

    def update(self, df):

        """
        Update statistics only after all features for the
        current day have been generated.
        """

        item_stats = (
            df.groupby("item_id")["clicked"]
              .agg(["size", "sum"])
        )

        for item, row in item_stats.iterrows():

            self.item_show[item] += int(row["size"])
            self.item_click[item] += int(row["sum"])

        user_stats = (
            df.groupby("user_id")["clicked"]
              .agg(["size", "sum"])
        )

        for user, row in user_stats.iterrows():

            self.user_show[user] += int(row["size"])
            self.user_click[user] += int(row["sum"])

        self.total_rows += len(df)

        self.total_clicks += int(
            df["clicked"].sum()
        )

    def copy(self):

        new = HistoricalStats()

        new.item_show.update(self.item_show)
        new.item_click.update(self.item_click)

        new.user_show.update(self.user_show)
        new.user_click.update(self.user_click)

        new.total_rows = self.total_rows
        new.total_clicks = self.total_clicks

        return new


# ============================================================
# Stream data by complete day
# ============================================================

def iter_daily_frames():

    """
    The input CSV must be ordered chronologically.

    Chunks are buffered until a complete date can be emitted.
    """

    pending = None

    reader = pd.read_csv(
        TRAIN_FILE,
        usecols=USE_COLUMNS,
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        chunk["date"] = pd.to_datetime(
            chunk["timestamp"],
            unit="s",
            utc=True
        ).dt.strftime("%Y-%m-%d")

        if pending is not None:

            chunk = pd.concat(
                [pending, chunk],
                ignore_index=True
            )

        dates = chunk["date"].drop_duplicates().tolist()

        if len(dates) == 1:

            pending = chunk
            continue

        for date in dates[:-1]:

            day_df = chunk[
                chunk["date"] == date
            ].copy()

            yield date, day_df

        last_date = dates[-1]

        pending = chunk[
            chunk["date"] == last_date
        ].copy()

    if pending is not None and len(pending):

        date = pending["date"].iloc[0]

        yield date, pending


# ============================================================
# History parsing
# ============================================================

def parse_history(value):

    if not isinstance(value, str):
        return []

    try:

        history = json.loads(value)

        if isinstance(history, list):
            return history

        return []

    except (json.JSONDecodeError, TypeError):

        return []


# ============================================================
# Feature generation
# ============================================================

def build_features(df, stats):

    features = []

    for row in df.itertuples(index=False):

        history = parse_history(
            row.history_item_ids
        )

        item = row.item_id
        user = row.user_id

        history_count = history.count(item)

        history_length = len(history)

        history_unique_count = len(
            set(history)
        )

        if history_length > 0:

            duplicate_ratio = (
                history_length - history_unique_count
            ) / history_length

        else:

            duplicate_ratio = 0.0

        features.append([

            stats.item_ctr(item),

            stats.user_ctr(user),

            int(history_count > 0),

            history_count,

            history_length,

            history_unique_count,

            duplicate_ratio,

            row.time_bucket,

            row.day_type,

        ])

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# PASS 1: Build leakage-controlled training features
# ============================================================

print("=" * 60)
print("PASS 1: Building chronological training features")
print("=" * 60)

stats = HistoricalStats()

daily_features = {}

stats_at_train_end = None

for date, day_df in iter_daily_frames():

    print(
        f"Processing {date}: "
        f"{len(day_df):,} rows"
    )

    # Generate features BEFORE incorporating today's labels.
    X_day = build_features(
        day_df,
        stats
    )

    y_day = day_df["clicked"].to_numpy(
        dtype=np.int8
    )

    daily_features[date] = (
        X_day,
        y_day
    )

    # Only now may today's labels enter historical statistics.
    stats.update(day_df)

    if date == TRAIN_END_DATE:

        stats_at_train_end = stats.copy()

    del day_df

    gc.collect()


if stats_at_train_end is None:

    raise RuntimeError(
        f"Could not find training cutoff date "
        f"{TRAIN_END_DATE}. Check timestamps and date parsing."
    )


all_dates = sorted(daily_features.keys())

train_dates = [
    d for d in all_dates
    if d <= TRAIN_END_DATE
]

valid_dates = [
    d for d in all_dates
    if VALID_START_DATE <= d <= VALID_END_DATE
]

if not train_dates or not valid_dates:

    raise RuntimeError(
        "Empty training or validation split. "
        "Check the date ranges."
    )


print("\nTraining dates:", train_dates)
print("Validation dates:", valid_dates)


# ============================================================
# Training matrix
# ============================================================

X_train = np.concatenate(
    [
        daily_features[d][0]
        for d in train_dates
    ],
    axis=0
)

y_train = np.concatenate(
    [
        daily_features[d][1]
        for d in train_dates
    ],
    axis=0
)


# ============================================================
# PASS 2: Validation features with frozen historical statistics
# ============================================================

print("\n" + "=" * 60)
print("PASS 2: Building leakage-controlled validation features")
print("=" * 60)

X_valid_parts = []
y_valid_parts = []

for date, day_df in iter_daily_frames():

    if date not in valid_dates:

        del day_df
        continue

    print(
        f"Validation features for {date}: "
        f"{len(day_df):,} rows"
    )

    # Freeze statistics at the end of January 24.
    # No labels from January 25 or 26 are used here.
    X_valid_day = build_features(
        day_df,
        stats_at_train_end
    )

    y_valid_day = day_df["clicked"].to_numpy(
        dtype=np.int8
    )

    X_valid_parts.append(X_valid_day)
    y_valid_parts.append(y_valid_day)

    del day_df

    gc.collect()


X_valid = np.concatenate(
    X_valid_parts,
    axis=0
)

y_valid = np.concatenate(
    y_valid_parts,
    axis=0
)


print("\nX_train:", X_train.shape)
print("X_valid:", X_valid.shape)

print(
    "Training row CTR:",
    float(y_train.mean())
)

print(
    "Validation row CTR:",
    float(y_valid.mean())
)


# ============================================================
# Train validation model
# ============================================================

print("\n" + "=" * 60)
print("Training LightGBM classifier")
print("=" * 60)

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

validation_model = lgb.LGBMClassifier(
    **params
)

validation_model.fit(

    X_train,
    y_train,

    eval_set=[
        (X_valid, y_valid)
    ],

    eval_metric="auc",

    callbacks=[

        lgb.early_stopping(
            stopping_rounds=100,
            verbose=True
        ),

        lgb.log_evaluation(
            period=50
        ),

    ],
)


valid_pred = validation_model.predict_proba(
    X_valid
)[:, 1]

valid_auc = roc_auc_score(
    y_valid,
    valid_pred
)

best_iteration = validation_model.best_iteration_

if not best_iteration:

    best_iteration = params["n_estimators"]


print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print("Validation AUC:", valid_auc)
print("Best iteration:", best_iteration)


# ============================================================
# Train final model on all training dates
# ============================================================

print("\n" + "=" * 60)
print("Training final model on all available dates")
print("=" * 60)

X_all = np.concatenate(
    [
        daily_features[d][0]
        for d in all_dates
    ],
    axis=0
)

y_all = np.concatenate(
    [
        daily_features[d][1]
        for d in all_dates
    ],
    axis=0
)

print("Final training matrix:", X_all.shape)

del X_train
del X_valid
del y_train
del y_valid

del X_valid_parts
del y_valid_parts
del daily_features
del validation_model
del valid_pred

gc.collect()


final_params = params.copy()

final_params["n_estimators"] = best_iteration

final_model = lgb.LGBMClassifier(
    **final_params
)

final_model.fit(
    X_all,
    y_all
)

final_model.booster_.save_model(
    MODEL_FILE
)

print("\nSaved model:", MODEL_FILE)
print("Final training rows:", len(y_all))
print("Validation AUC:", valid_auc)
print("Done.")