import pandas as pd

train_users = set()
train_items = set()

print("reading train...")

for chunk in pd.read_csv(
        "train.csv",
        chunksize=100000):

    train_users.update(
        chunk["user_id"]
    )

    train_items.update(
        chunk["item_id"]
    )

print()

print(
    "train users:",
    len(train_users)
)

print(
    "train items:",
    len(train_items)
)

eval_rows = 0

seen_users = 0
seen_items = 0

print()
print("reading evaluation...")

for chunk in pd.read_csv(
        "evaluation.csv",
        chunksize=100000):

    eval_rows += len(chunk)

    for u in chunk["user_id"]:

        if u in train_users:
            seen_users += 1

    for i in chunk["item_id"]:

        if i in train_items:
            seen_items += 1

print()

print(
    "evaluation rows:",
    eval_rows
)

print()

print(
    "user coverage:",
    seen_users / eval_rows
)

print(
    "item coverage:",
    seen_items / eval_rows
)