import pandas as pd
import json
import numpy as np
from collections import defaultdict
from scipy.special import softmax

TRAIN_FILE = "train.csv"
EVAL_FILE = "evaluation.csv"

print("Pass 1: collecting CTR statistics...")

item_show = defaultdict(int)
item_click = defaultdict(int)

user_show = defaultdict(int)
user_click = defaultdict(int)

chunksize = 100000

for chunk in pd.read_csv(
    TRAIN_FILE,
    chunksize=chunksize
):

    for row in chunk.itertuples(index=False):

        item_show[row.item_id] += 1
        user_show[row.user_id] += 1

        if row.clicked == 1:
            item_click[row.item_id] += 1
            user_click[row.user_id] += 1

print("Statistics collected.")

GLOBAL_CTR = sum(item_click.values()) / sum(item_show.values())

print("Global CTR =", GLOBAL_CTR)

def item_ctr(item):

    c = item_click[item]
    n = item_show[item]

    return (c + 1) / (n + 10)

def user_ctr(user):

    c = user_click[user]
    n = user_show[user]

    if n == 0:
        return GLOBAL_CTR

    return (c + 1) / (n + 10)

print("Generating predictions...")

predictions = []

for chunk in pd.read_csv(
    EVAL_FILE,
    chunksize=50000
):

    scores = []

    for row in chunk.itertuples(index=False):

        item_score = item_ctr(row.item_id)

        user_score = user_ctr(row.user_id)

        history = json.loads(row.history_item_ids)

        repeat_click = int(row.item_id in history)

        history_count = history.count(row.item_id)

        score = (
            0.50 * item_score
            + 0.30 * user_score
            + 0.15 * repeat_click
            + 0.05 * history_count
        )

        scores.append(score)

    chunk["raw_score"] = scores

    for event_id, group in chunk.groupby("event_id"):

        probs = softmax(group["raw_score"].values)

        for rowid, p in zip(group["rowid"], probs):

            predictions.append(
                (rowid, float(p))
            )

submission = pd.DataFrame(
    predictions,
    columns=[
        "rowid",
        "predicted_probability"
    ]
)

submission.to_csv(
    "submission.csv",
    index=False
)

print("Done.")
print("submission.csv created.")