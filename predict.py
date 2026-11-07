import json
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = "train.csv"
EVAL_FILE = "evaluation.csv"

MODEL_FILE = "ctr_model_improved.txt"

OUTPUT_FILE = "submission_improved.csv"

CHUNK_SIZE = 200_000

SMOOTHING = 50.0
PRIOR_CTR = 0.06

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


# ============================================================
# Stream training data by day
# ============================================================

def iter_daily_frames():

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

            yield date, chunk[
                chunk["date"] == date
            ].copy()

        last_date = dates[-1]

        pending = chunk[
            chunk["date"] == last_date
        ].copy()

    if pending is not None and len(pending):

        yield pending["date"].iloc[0], pending


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
# Build statistics using training data only
# ============================================================

print("Loading training statistics...")

stats = HistoricalStats()

for date, day_df in iter_daily_frames():

    stats.update(day_df)

    print(
        f"Processed {date}: "
        f"{len(day_df):,} rows"
    )

    del day_df

    gc.collect()


print("Global CTR:", stats.global_ctr())


# ============================================================
# Load model
# ============================================================

model = lgb.Booster(
    model_file=MODEL_FILE
)

print("Model loaded:", MODEL_FILE)


# ============================================================
# Predict evaluation data
# ============================================================

rowids = []
predictions = []

print("Generating predictions...")

for chunk in pd.read_csv(
    EVAL_FILE,
    chunksize=CHUNK_SIZE
):

    X = build_features(
        chunk,
        stats
    )

    # Binary LightGBM returns probabilities by default.
    # Do not apply event-level Softmax.
    scores = model.predict(X)

    rowids.extend(
        chunk["rowid"].tolist()
    )

    predictions.extend(
        scores.tolist()
    )

    print(
        "Predicted rows:",
        len(rowids)
    )

    del X
    del chunk

    gc.collect()


# ============================================================
# Save submission
# ============================================================

submission = pd.DataFrame({

    "rowid": rowids,

    "predicted_probability": predictions,

})

submission.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSubmission generated:", OUTPUT_FILE)
print("Total rows:", len(submission))
print("Prediction range:")
print(submission["predicted_probability"].describe())