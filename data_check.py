# data_check.py
import pandas as pd
import glob
import os

# ------------------------------------------------------
# 1. raw/ 配下の全フレームCSVを読み込み、session_idを付与
# ------------------------------------------------------
raw_files = sorted(glob.glob("data/raw/*_frames.csv"))
print(f"📁 セッション数: {len(raw_files)}")

df_list = []
for f in raw_files:
    df = pd.read_csv(f)
    # ファイル名からsession_idを抽出（UUID部分）
    session_id = os.path.basename(f).replace("_frames.csv", "")
    df["session_id"] = session_id
    df_list.append(df)

all_df = pd.concat(df_list, ignore_index=True)

print(f"🎞️ 総フレーム数: {len(all_df)}")

# phase, session_id, frame_index, timestamp_ms 以外が特徴量
non_feature_cols = ["frame_index", "timestamp_ms", "phase", "session_id"]
feature_cols = [c for c in all_df.columns if c not in non_feature_cols]
print(f"🧮 特徴量数: {len(feature_cols)}")

# ------------------------------------------------------
# 2. phase（実質ラベル）別のサンプル数
# ------------------------------------------------------
print("\n🔖 phase別サンプル数:")
print(all_df["phase"].value_counts())

# ------------------------------------------------------
# 4. 集計済みファイルも確認
# ------------------------------------------------------
for path in ["data/aggregated/sessions.csv", "data/sessions/summary.csv"]:
    if os.path.exists(path):
        print(f"\n📄 {path} のカラム:")
        print(list(pd.read_csv(path).columns))