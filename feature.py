import pandas as pd

TRAIN_PATH = "train.csv"

df = pd.read_csv(
    TRAIN_PATH,
    usecols=[
        "event_id",
        "user_id",
        "item_id",
        "timestamp",
        "clicked"
    ]
)

df["date"] = pd.to_datetime(
    df["timestamp"],
    unit="s",
    utc=True
).dt.strftime("%Y-%m-%d")

# 每个事件是否至少有一个点击
event_stats = (
    df.groupby(["date", "event_id"])["clicked"]
      .max()
      .groupby("date")
      .agg(
          total_events="count",
          events_with_click="sum"
      )
)

event_stats["event_click_rate"] = (
    event_stats["events_with_click"]
    / event_stats["total_events"]
)

# 每天的用户和物品数量
daily_users = df.groupby("date")["user_id"].nunique()
daily_items = df.groupby("date")["item_id"].nunique()

event_stats["unique_users"] = daily_users
event_stats["unique_items"] = daily_items

print(event_stats.to_string())