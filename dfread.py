# 簡易チェック用スニペット
import pandas as pd
sessions = pd.read_csv(r"C:\Users\masaa\Documents\Python\KIKAGAKU\eq_gym_poc\data\aggregated\sessions.csv")
print(sessions["baseline_id"].value_counts())  # baseline紐づけ状況
print(sessions["baseline_id"].isna().sum())     # 未紐づけ件数

sessions = pd.read_csv(r"C:\Users\masaa\Documents\Python\KIKAGAKU\eq_gym_poc\data\aggregated\sessions.csv")
baselines = pd.read_csv(r"C:\Users\masaa\Documents\Python\KIKAGAKU\eq_gym_poc\data\aggregated\baseline_scores.csv")
missing = set(sessions["baseline_id"].dropna()) - set(baselines["baseline_id"])
print(f"欠落しているbaseline_id: {missing}")
print(baselines[["baseline_id", "date", "recorded_at", "is_manual"]])

# 🔍 事前診断スクリプト（実行推奨）
import pandas as pd

df_s = pd.read_csv(r"C:\Users\masaa\Documents\Python\KIKAGAKU\eq_gym_poc\data\aggregated\sessions.csv")
df_b = pd.read_csv(r"C:\Users\masaa\Documents\Python\KIKAGAKU\eq_gym_poc\data\aggregated\baseline_scores.csv")

valid_ids = set(df_b["baseline_id"])
linked = df_s["baseline_id"].notna() & df_s["baseline_id"].isin(valid_ids)

print(f"総セッション数: {len(df_s)}")
print(f"baseline紐づけ有効: {linked.sum()}")
print(f"baseline_id欠損: {df_s['baseline_id'].isna().sum()}")
print(f"baseline_id存在するが参照先なし: {(df_s['baseline_id'].notna() & ~df_s['baseline_id'].isin(valid_ids)).sum()}")

import glob, os, pandas as pd
from data_recorder import RAW_DIR

df_b = pd.read_csv(r"C:\Users\masaa\Documents\Python\KIKAGAKU\eq_gym_poc\data\aggregated\baseline_scores.csv")
meta = ["baseline_id", "date", "recorded_at", "is_manual"]
baseline_feature_cols = [c for c in df_b.columns if c not in meta]

sample_path = glob.glob(os.path.join(str(RAW_DIR), "*_frames.csv"))[0]
sample = pd.read_csv(sample_path)

missing = set(baseline_feature_cols) - set(sample.columns)
print(f"baseline側:{len(baseline_feature_cols)}列 / raw側に無い列: {missing}")

print("phaseユニーク値:", sample["phase"].unique())

import joblib
bundle = joblib.load("phase_classifier_rf_delta.pkl")
print(bundle["model"].classes_)