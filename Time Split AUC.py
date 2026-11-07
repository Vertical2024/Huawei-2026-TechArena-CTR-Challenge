import pandas as pd
import numpy as np
import json

from collections import defaultdict
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

TRAIN_FILE = "train.csv"

chunksize = 100000

item_show = defaultdict(int)
item_click = defaultdict(int)

user_show = defaultdict(int)
user_click = defaultdict(int)

total_clicks = 0
total_rows = 0

print("PASS 1")

for chunk in pd.read_csv(
        TRAIN_FILE,
        chunksize=chunksize):

    for row in chunk.itertuples(index=False):

        item_show[row.item_id] += 1
        user_show[row.user_id] += 1

        if row.clicked:

            item_click[row.item_id] += 1
            user_click[row.user_id] += 1

            total_clicks += 1

        total_rows += 1

global_ctr = total_clicks / total_rows

print("global ctr:", global_ctr)

def item_ctr(item):

    return (
        item_click[item]
        + 50 * global_ctr
    ) / (
        item_show[item]
        + 50
    )

def user_ctr(user):

    return (
        user_click[user]
        + 50 * global_ctr
    ) / (
        user_show[user]
        + 50
    )

print("PASS 2")

features = []
labels = []
timestamps = []

for chunk in pd.read_csv(
        TRAIN_FILE,
        chunksize=50000):

    for row in chunk.itertuples(index=False):

        history = json.loads(
            row.history_item_ids
        )

        repeat_click = int(
            row.item_id in history
        )

        history_count = history.count(
            row.item_id
        )

        features.append([

            item_ctr(row.item_id),
            user_ctr(row.user_id),

            repeat_click,
            history_count,

            len(history),

            row.time_bucket,
            row.day_type
        ])

        labels.append(
            row.clicked
        )

        timestamps.append(
            row.timestamp
        )

X = np.array(
    features,
    dtype=np.float32
)

y = np.array(labels)

timestamps = np.array(
    timestamps
)

cutoff = np.quantile(
    timestamps,
    0.85
)

print("cutoff =", cutoff)

train_mask = (
    timestamps < cutoff
)

valid_mask = (
    timestamps >= cutoff
)

X_train = X[train_mask]
y_train = y[train_mask]

X_valid = X[valid_mask]
y_valid = y[valid_mask]

print("train size:", len(X_train))
print("valid size:", len(X_valid))

model = lgb.LGBMClassifier(

    objective="binary",

    n_estimators=500,

    learning_rate=0.05,

    num_leaves=63,

    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict_proba(
    X_valid
)[:, 1]

auc = roc_auc_score(
    y_valid,
    pred
)

print()
print("TIME SPLIT AUC =", auc)