# lasso_feature_selector.py
# Lasso回帰で個人の有効特徴量を選択し、重みを返す

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

from calibration import ALL_FEATURES, compute_baseline, compute_delta

# 対象フェーズ（それぞれをbinaryで分類）
TARGET_PHASES = ["angry", "smile", "brow_up"]


# ============================================================
# メイン: セッションCSVからLasso重みを計算
# ============================================================
def compute_lasso_weights(
    df: pd.DataFrame,
    min_samples_per_phase: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    セッションCSVデータからフェーズごとのLasso重みを計算する

    Args:
        df: CSVから読み込んだDataFrame
            必須列: phase, + ALL_FEATURES
        min_samples_per_phase: 最低必要フレーム数

    Returns:
        {
            "angry":   {"browDownRight": 0.42, "mouthPressLeft": 0.38, ...},
            "smile":   {"eyeSquintLeft": 0.55, ...},
            "brow_up": {"browInnerUp": 0.61, ...}
        }
        ※ 重みが0の特徴量はdictに含まれない（Lassoで除外された）
    """
    # --- Step1: neutralベースライン計算 ---
    neutral_df = df[df["phase"] == "neutral"]
    if len(neutral_df) < min_samples_per_phase:
        print("[lasso] ⚠️ neutralフレームが少ない。ゼロベースラインを使用")
        baseline = {feat: 0.0 for feat in ALL_FEATURES}
    else:
        neutral_frames = neutral_df[ALL_FEATURES].to_dict(orient="records")
        baseline = compute_baseline(neutral_frames)
        print(f"[lasso] ✅ neutralベースライン計算完了 ({len(neutral_df)}フレーム)")

    # --- Step2: 全フレームにデルタ適用 ---
    delta_records = []
    for _, row in df.iterrows():
        frame = row[ALL_FEATURES].to_dict()
        delta = compute_delta(frame, baseline)
        delta["phase"] = row["phase"]
        delta_records.append(delta)
    delta_df = pd.DataFrame(delta_records)

    # --- Step3: フェーズごとにLasso ---
    lasso_weights = {}

    for target_phase in TARGET_PHASES:
        print(f"\n[lasso] 🔍 フェーズ '{target_phase}' のLasso計算中...")

        # 対象フェーズ vs neutral の2クラス
        mask = delta_df["phase"].isin([target_phase, "neutral"])
        sub_df = delta_df[mask].copy()

        if len(sub_df[sub_df["phase"] == target_phase]) < min_samples_per_phase:
            print(f"[lasso]   ⚠️ '{target_phase}'フレームが不足。スキップ")
            lasso_weights[target_phase] = {}
            continue

        # 特徴量行列 X とラベル y
        X = sub_df[ALL_FEATURES].values
        y = (sub_df["phase"] == target_phase).astype(float).values

        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # LassoCV（自動でalphaを選択）
        lasso = LassoCV(
            cv=5,
            max_iter=5000,
            random_state=42,
            alphas=np.logspace(-4, 1, 50)
        )
        lasso.fit(X_scaled, y)

        # 重みを正規化（0以外のみ）
        raw_weights = dict(zip(ALL_FEATURES, lasso.coef_))
        active_weights = {
            feat: float(w)
            for feat, w in raw_weights.items()
            if abs(w) > 1e-6  # Lassoで0になったものを除外
        }

        # 正の重みのみ（そのフェーズで増加する特徴量）を残す
        positive_weights = {
            feat: w for feat, w in active_weights.items() if w > 0
        }

        # 合計1.0に正規化
        total = sum(positive_weights.values())
        if total > 0:
            normalized = {
                feat: round(w / total, 4)
                for feat, w in positive_weights.items()
            }
        else:
            normalized = {}

        lasso_weights[target_phase] = normalized

        # 結果表示
        print(f"[lasso]   選択された特徴量 ({len(normalized)}個):")
        for feat, w in sorted(normalized.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 20)
            print(f"    {feat:<22}: {w:.4f} {bar}")
        print(f"[lasso]   最適alpha = {lasso.alpha_:.6f}")

    return lasso_weights, baseline


# ============================================================
# 実行例（単体テスト用）
# ============================================================
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
               "data/raw_csv/all_frames_combined.csv"

    print(f"[lasso] 📂 読み込み: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[lasso] フレーム数: {len(df)}, フェーズ: {df['phase'].unique()}")

    weights, baseline = compute_lasso_weights(df)

    print("\n" + "=" * 60)
    print("📊 最終結果サマリー")
    print("=" * 60)
    for phase, w_dict in weights.items():
        print(f"\n  [{phase}]")
        if w_dict:
            for feat, w in sorted(w_dict.items(), key=lambda x: -x[1]):
                print(f"    {feat:<22}: {w:.4f}")
        else:
            print("    (特徴量なし)")