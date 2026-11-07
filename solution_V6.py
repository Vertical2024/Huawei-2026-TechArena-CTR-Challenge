import gc
import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

# ============================================================
# Global Configuration
# ============================================================
TRAIN_FILE = "train.csv"
EVAL_FILE = "evaluation.csv"
SUBMISSION_FILE = "submission_v6.csv"

TRAIN_END_DATE = "2025-01-24"
VALID_START_DATE = "2025-01-25"
VALID_END_DATE = "2025-01-26"

SMOOTHING = 50.0
PRIOR_CTR = 0.06
W2V_DIM = 32

USE_COLS_TRAIN = [
    "rowid", "event_id", "user_id", "item_id", "timestamp",
    "time_bucket", "day_type", "history_item_ids", "history_timestamps", "clicked"
]
USE_COLS_EVAL = [
    "rowid", "event_id", "user_id", "item_id", "timestamp",
    "time_bucket", "day_type", "history_item_ids", "history_timestamps"
]

def fast_parse_str_array(s):
    if not isinstance(s, str) or len(s) <= 2 or s == "[]":
        return []
    return [x.strip(' "\'') for x in s[1:-1].split(',')]

def fast_parse_int_array(s):
    if not isinstance(s, str) or len(s) <= 2 or s == "[]":
        return []
    try:
        return [int(x.strip(' "\'')) for x in s[1:-1].split(',')]
    except ValueError:
        return []

def train_item2vec(df_all_hist):
    print("\n[Step 1/4] Training Item2Vec (Word2Vec)...")
    sentences = [fast_parse_str_array(s) for s in df_all_hist["history_item_ids"].dropna().unique() if len(s) > 2]
    if HAS_GENSIM and len(sentences) > 0:
        w2v_model = Word2Vec(sentences=sentences, vector_size=W2V_DIM, window=5, min_count=1, workers=8, epochs=10, seed=42)
        print("Word2Vec model successfully trained.")
        return w2v_model.wv
    return None

class V6FeatureExtractor:
    def __init__(self, item_stats, item_tb_stats, user_stats, global_total, global_clicks, w2v_vectors):
        self.item_show = item_stats.get("show", {})
        self.item_click = item_stats.get("click", {})
        
        self.item_tb_show = item_tb_stats.get("show", {})
        self.item_tb_click = item_tb_stats.get("click", {})
        
        self.user_show = user_stats.get("show", {})
        self.user_click = user_stats.get("click", {})
        
        self.global_ctr = (global_clicks / global_total) if global_total > 0 else PRIOR_CTR
        self.w2v = w2v_vectors

    def get_item_ctr(self, item):
        shows = self.item_show.get(item, 0)
        clicks = self.item_click.get(item, 0)
        return (clicks + SMOOTHING * self.global_ctr) / (shows + SMOOTHING)

    def get_item_tb_ctr(self, item, tb):
        key = (item, tb)
        shows = self.item_tb_show.get(key, 0)
        clicks = self.item_tb_click.get(key, 0)
        return (clicks + SMOOTHING * self.global_ctr) / (shows + SMOOTHING)

    def get_user_ctr(self, user):
        shows = self.user_show.get(user, 0)
        clicks = self.user_click.get(user, 0)
        return (clicks + SMOOTHING * self.global_ctr) / (shows + SMOOTHING)

    def extract_features(self, df):
        n_rows = len(df)
        item_ctrs = np.zeros(n_rows, dtype=np.float32)
        item_tb_ctrs = np.zeros(n_rows, dtype=np.float32)
        user_ctrs = np.zeros(n_rows, dtype=np.float32)
        item_shows = np.zeros(n_rows, dtype=np.float32)
        
        hist_lens = np.zeros(n_rows, dtype=np.float32)
        hist_uniques = np.zeros(n_rows, dtype=np.float32)
        hist_click_counts = np.zeros(n_rows, dtype=np.float32)
        is_in_hist = np.zeros(n_rows, dtype=np.int8)
        is_last_clicked = np.zeros(n_rows, dtype=np.int8)
        is_in_last_3 = np.zeros(n_rows, dtype=np.int8)
        last_click_pos = np.full(n_rows, -1, dtype=np.float32)
        
        log_time_since_user_last_click = np.full(n_rows, -1.0, dtype=np.float32)
        log_time_since_item_last_click = np.full(n_rows, -1.0, dtype=np.float32)
        
        w2v_sim_mean = np.zeros(n_rows, dtype=np.float32)
        w2v_sim_last = np.zeros(n_rows, dtype=np.float32)
        w2v_sim_max = np.zeros(n_rows, dtype=np.float32)
        
        item_ids = df["item_id"].to_numpy()
        user_ids = df["user_id"].to_numpy()
        timestamps = df["timestamp"].to_numpy()
        time_buckets = df["time_bucket"].to_numpy()
        hist_items_str = df["history_item_ids"].to_numpy()
        hist_times_str = df["history_timestamps"].to_numpy()
        
        for idx in range(n_rows):
            item = item_ids[idx]
            user = user_ids[idx]
            ts = timestamps[idx]
            tb = time_buckets[idx]
            
            item_ctrs[idx] = self.get_item_ctr(item)
            item_tb_ctrs[idx] = self.get_item_tb_ctr(item, tb)
            user_ctrs[idx] = self.get_user_ctr(user)
            item_shows[idx] = self.item_show.get(item, 0)
            
            h_items = fast_parse_str_array(hist_items_str[idx])
            h_times = fast_parse_int_array(hist_times_str[idx])
            h_len = len(h_items)
            hist_lens[idx] = h_len
            
            if h_len > 0:
                hist_uniques[idx] = len(set(h_items))
                item_count = h_items.count(item)
                hist_click_counts[idx] = item_count
                
                if item_count > 0:
                    is_in_hist[idx] = 1
                    rev_pos = h_items[::-1].index(item)
                    pos = h_len - 1 - rev_pos
                    last_click_pos[idx] = pos
                    if len(h_times) == h_len:
                        diff_sec = max(0, ts - h_times[pos])
                        log_time_since_item_last_click[idx] = np.log1p(diff_sec)
                
                is_last_clicked[idx] = int(item == h_items[-1])
                is_in_last_3[idx] = int(item in h_items[-3:])
                if len(h_times) == h_len:
                    diff_sec = max(0, ts - h_times[-1])
                    log_time_since_user_last_click[idx] = np.log1p(diff_sec)

                if self.w2v is not None and item in self.w2v:
                    cand_vec = self.w2v[item]
                    cand_norm = np.linalg.norm(cand_vec)
                    if cand_norm > 0:
                        hist_vecs = [self.w2v[h] for h in h_items if h in self.w2v]
                        if hist_vecs:
                            m_vec = np.mean(hist_vecs, axis=0)
                            m_norm = np.linalg.norm(m_vec)
                            if m_norm > 0:
                                sim = float(np.dot(cand_vec, m_vec) / (cand_norm * m_norm + 1e-8))
                                w2v_sim_mean[idx] = max(0.0, 0.5 * (sim + 1.0))
                            
                            last_item = h_items[-1]
                            if last_item in self.w2v:
                                l_vec = self.w2v[last_item]
                                l_norm = np.linalg.norm(l_vec)
                                if l_norm > 0:
                                    sim = float(np.dot(cand_vec, l_vec) / (cand_norm * l_norm + 1e-8))
                                    w2v_sim_last[idx] = max(0.0, 0.5 * (sim + 1.0))
                            
                            sims = [0.5 * (np.dot(cand_vec, hv) / (cand_norm * np.linalg.norm(hv) + 1e-8) + 1.0) for hv in hist_vecs if np.linalg.norm(hv) > 0]
                            if sims:
                                w2v_sim_max[idx] = max(0.0, float(max(sims)))

        feat_df = pd.DataFrame({
            "event_id": df["event_id"],
            "item_ctr": item_ctrs,
            "item_tb_ctr": item_tb_ctrs,
            "user_ctr": user_ctrs,
            "item_shows": item_shows,
            "hist_len": hist_lens,
            "hist_uniques": hist_uniques,
            "hist_click_counts": hist_click_counts,
            "is_in_hist": is_in_hist,
            "is_last_clicked": is_last_clicked,
            "is_in_last_3": is_in_last_3,
            "last_click_pos": last_click_pos,
            "log_time_since_user_last_click": log_time_since_user_last_click,
            "log_time_since_item_last_click": log_time_since_item_last_click,
            "w2v_sim_mean": w2v_sim_mean,
            "w2v_sim_last": w2v_sim_last,
            "w2v_sim_max": w2v_sim_max,
            "time_bucket": df["time_bucket"].astype(np.int8),
            "day_type": df["day_type"].astype(np.int8)
        })
        
        g = feat_df.groupby("event_id")
        for col in ["item_ctr", "item_tb_ctr", "w2v_sim_mean", "w2v_sim_last", "hist_click_counts"]:
            feat_df[f"{col}_rank_in_event"] = g[col].rank(ascending=False, method="min").astype(np.float32)
            feat_df[f"{col}_diff_mean_event"] = (feat_df[col] - g[col].transform("mean")).astype(np.float32)
            feat_df[f"{col}_diff_max_event"] = (feat_df[col] - g[col].transform("max")).astype(np.float32)
            feat_df[f"{col}_ratio_sum_event"] = (feat_df[col] / (g[col].transform("sum") + 1e-6)).astype(np.float32)
            
        feat_cols = [c for c in feat_df.columns if c != "event_id"]
        return feat_df[feat_cols].to_numpy(dtype=np.float32), feat_cols

def main():
    start_time = time.time()
    print("=" * 70)
    print("TechArena 2026 Phase 1: V6 Calibrated Baseline Pipeline")
    print("=" * 70)
    
    print("\n[Step 2/4] Loading datasets into RAM...")
    df_train = pd.read_csv(TRAIN_FILE, usecols=USE_COLS_TRAIN)
    df_eval = pd.read_csv(EVAL_FILE, usecols=USE_COLS_EVAL)
    
    df_train["date"] = pd.to_datetime(df_train["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    df_train["clicked"] = df_train["clicked"].astype(np.int8)
    
    df_all_hist = pd.concat([df_train[["history_item_ids"]], df_eval[["history_item_ids"]]], ignore_index=True)
    w2v_vectors = train_item2vec(df_all_hist)
    del df_all_hist
    gc.collect()

    print("\n[Step 3/4] Building leakage-controlled historical statistics...")
    train_mask = df_train["date"] <= TRAIN_END_DATE
    valid_mask = (df_train["date"] >= VALID_START_DATE) & (df_train["date"] <= VALID_END_DATE)
    
    df_train_sub = df_train[train_mask]
    df_valid_sub = df_train[valid_mask]

    # Validation Stats (<= Jan 24)
    item_agg = df_train_sub.groupby("item_id")["clicked"].agg(["size", "sum"])
    item_tb_agg = df_train_sub.groupby(["item_id", "time_bucket"])["clicked"].agg(["size", "sum"])
    user_agg = df_train_sub.groupby("user_id")["clicked"].agg(["size", "sum"])
    
    val_item_stats = {"show": item_agg["size"].to_dict(), "click": item_agg["sum"].to_dict()}
    val_item_tb_stats = {"show": item_tb_agg["size"].to_dict(), "click": item_tb_agg["sum"].to_dict()}
    val_user_stats = {"show": user_agg["size"].to_dict(), "click": user_agg["sum"].to_dict()}
    val_global_total = len(df_train_sub)
    val_global_clicks = int(df_train_sub["clicked"].sum())

    # Full Stats (All 7 days)
    item_agg_full = df_train.groupby("item_id")["clicked"].agg(["size", "sum"])
    item_tb_agg_full = df_train.groupby(["item_id", "time_bucket"])["clicked"].agg(["size", "sum"])
    user_agg_full = df_train.groupby("user_id")["clicked"].agg(["size", "sum"])
    
    full_item_stats = {"show": item_agg_full["size"].to_dict(), "click": item_agg_full["sum"].to_dict()}
    full_item_tb_stats = {"show": item_tb_agg_full["size"].to_dict(), "click": item_tb_agg_full["sum"].to_dict()}
    full_user_stats = {"show": user_agg_full["size"].to_dict(), "click": user_agg_full["sum"].to_dict()}
    full_global_total = len(df_train)
    full_global_clicks = int(df_train["clicked"].sum())

    print("\n[Step 4/4] Extracting validation matrices...")
    extractor_val = V6FeatureExtractor(val_item_stats, val_item_tb_stats, val_user_stats, val_global_total, val_global_clicks, w2v_vectors)
    X_tr, feature_names = extractor_val.extract_features(df_train_sub)
    y_tr = df_train_sub["clicked"].to_numpy(dtype=np.int8)
    
    X_va, _ = extractor_val.extract_features(df_valid_sub)
    y_va = df_valid_sub["clicked"].to_numpy(dtype=np.int8)

    print(f"Features dimension: {X_tr.shape[1]} features")

    # Train Validation Models
    print("\nTraining Validation LightGBM...")
    lgb_params = {
        "objective": "binary", "eval_metric": "auc", "n_estimators": 2500,
        "learning_rate": 0.04, "num_leaves": 127, "max_depth": -1, "min_child_samples": 80,
        "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0,
        "random_state": 42, "n_jobs": -1, "verbose": -1
    }
    lgb_val = lgb.LGBMClassifier(**lgb_params)
    lgb_val.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    val_pred_lgb = lgb_val.predict_proba(X_va)[:, 1]
    best_iter = lgb_val.best_iteration_
    print(f"Validation LightGBM AUC: {roc_auc_score(y_va, val_pred_lgb):.6f}")

    print("Training Validation CatBoost...")
    cb_val = CatBoostClassifier(iterations=1800, learning_rate=0.05, depth=7, eval_metric="AUC", random_seed=42, thread_count=-1, verbose=0)
    cb_val.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
    val_pred_cb = cb_val.predict_proba(X_va)[:, 1]
    print(f"Validation CatBoost AUC: {roc_auc_score(y_va, val_pred_cb):.6f}")

    val_pred_blend = 0.5 * val_pred_lgb + 0.5 * val_pred_cb
    auc_blend = roc_auc_score(y_va, val_pred_blend)
    print(f"\n>>>> V6 BLENDED VALIDATION ROC AUC: {auc_blend:.6f} <<<<")

    del X_tr, y_tr, X_va, y_va, lgb_val, cb_val
    gc.collect()

    # Retrain Final Models
    print("\nRetraining final models on full dataset...")
    extractor_full = V6FeatureExtractor(full_item_stats, full_item_tb_stats, full_user_stats, full_global_total, full_global_clicks, w2v_vectors)
    
    X_full, _ = extractor_full.extract_features(df_train)
    y_full = df_train["clicked"].to_numpy(dtype=np.int8)
    X_eval, _ = extractor_full.extract_features(df_eval)
    
    print("Fitting Final LightGBM...")
    final_lgb = lgb.LGBMClassifier(**lgb_params)
    final_lgb.n_estimators = int(best_iter * 1.15)
    final_lgb.fit(X_full, y_full)
    eval_pred_lgb = final_lgb.predict_proba(X_eval)[:, 1]

    print("Fitting Final CatBoost...")
    final_cb = CatBoostClassifier(iterations=1800, learning_rate=0.05, depth=7, eval_metric="AUC", random_seed=42, thread_count=-1, verbose=0)
    final_cb.fit(X_full, y_full)
    eval_pred_cb = final_cb.predict_proba(X_eval)[:, 1]

    # Directly blend raw probabilities and clip to strictly [0, 1]
    final_preds = np.clip(0.5 * eval_pred_lgb + 0.5 * eval_pred_cb, 0.0, 1.0)
    
    sub_df = pd.DataFrame({"rowid": df_eval["rowid"], "predicted_probability": final_preds})
    sub_df.to_csv(SUBMISSION_FILE, index=False)
    
    print("\n" + "=" * 70)
    print(f"V6 SUCCESS! Submission saved to: {SUBMISSION_FILE}")
    print(f"Validation Blended AUC: {auc_blend:.6f}")
    print(f"Pipeline Runtime: {(time.time() - start_time)/60:.2f} mins")
    print("=" * 70)

if __name__ == "__main__":
    main()