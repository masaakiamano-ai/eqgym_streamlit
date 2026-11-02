# ============================================================
# ml_direction_model.py
# 概要: 学習済み phase_classifier_rf_delta.pkl を使い、
#       calc_direction_scores() と同一シグネチャで
#       4方向スコア(DirectionScores)を返すMLラッパー
#
# 前提（学習時と厳密に一致させる必要あり）:
# - delta = current - baseline（クリップなし、負値もそのまま使用）
# - 各フレーム独立予測（時系列モデルではない）
# - feature_cols / selected_idx / scaler は pkl から動的取得
#
# 出力:
# - RandomForestClassifier.predict_proba() の4クラス確率を
#   phase名 → 方向軸名 にマッピングして DirectionScores として返す
# ============================================================
import joblib
import numpy as np
import streamlit as st
from emotion_direction import BlendshapeFrame, DirectionScores

MODEL_PATH = "phase_classifier_rf_delta.pkl"

# phase(分類クラス) → 方向軸 のマッピング
# app_main.py の PHASES[].target_axis と対応させている
PHASE_TO_AXIS = {
    "smile":   "positive",
    "brow_up": "tension",
    "angry":   "anger",
    "neutral": "stable",
}

# ── 将来的なスムージング拡張用フラグ（現状は未使用） ─────────
# フリーズ等の問題が確認された場合、compute_smoothed_scores()側で
# EMA処理を追加検討する。現時点では生の出力をそのまま返す方針。
ENABLE_ML_SMOOTHING = False


@st.cache_resource
def get_direction_model_bundle() -> dict:
    """学習済みモデルバンドルをロードしてキャッシュする"""
    bundle = joblib.load(MODEL_PATH)
    required_keys = {"model", "scaler", "selected_idx", "feature_cols"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise KeyError(
            f"モデルバンドルに必要なキーが不足しています: {missing}"
        )
    return bundle


def _extract_delta_vector(
    current: BlendshapeFrame,
    baseline: BlendshapeFrame,
    feature_cols: list,
) -> np.ndarray:
    """
    学習時と同一の前処理でdeltaベクトルを生成する。

    重要:
    - クリップなし（負値もそのまま）
    - feature_cols の順序を厳守（学習時と完全一致させる）
    """
    vec = np.array(
        [current.get(k, 0.0) - baseline.get(k, 0.0) for k in feature_cols],
        dtype=float,
    )
    return vec.reshape(1, -1)


def predict_direction_scores(
    current_frame: BlendshapeFrame,
    baseline_frame: "BlendshapeFrame | None",
) -> DirectionScores:
    """
    calc_direction_scores() と同一シグネチャ・同一戻り値型。

    baseline_frame が None の場合は ValueError を送出する。
    （呼び出し側 calc_direction_scores() が try/except で
      rule-based計算に自動フォールバックする設計とセットで使うこと）

    Raises:
        ValueError: baseline_frame が None の場合
        KeyError  : モデルバンドルの構造が不正な場合
    """
    if baseline_frame is None:
        raise ValueError(
            "predict_direction_scores: baseline_frame is None. "
            "呼び出し側でフォールバック処理をしてください。"
        )

    bundle = get_direction_model_bundle()
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    selected_idx = bundle["selected_idx"]
    feature_cols = bundle["feature_cols"]

    # 1. delta生成（クリップなし・学習時と同じ順序）
    delta_vec = _extract_delta_vector(current_frame, baseline_frame, feature_cols)

    # 2. スケーリング（全feature_colsに対して。学習時と同じscaler）
    delta_scaled = scaler.transform(delta_vec)

    # 3. Lassoで選択済みの特徴量のみ抽出
    X_selected = delta_scaled[:, selected_idx]

    # 4. 分類確率取得
    proba = model.predict_proba(X_selected)[0]
    class_names = model.classes_

    # 5. クラス確率 → 方向軸スコアへマッピング
    axis_scores = {"tension": 0.0, "anger": 0.0, "positive": 0.0, "stable": 0.0}
    for cls_name, p in zip(class_names, proba):
        axis = PHASE_TO_AXIS.get(cls_name)
        if axis is not None:
            axis_scores[axis] = float(p)

    # predict_proba の合計は必ず1.0なので、そのまま比率として使える
    return DirectionScores(
        tension=axis_scores["tension"],
        anger=axis_scores["anger"],
        positive=axis_scores["positive"],
        stable=axis_scores["stable"],
        ratio=axis_scores.copy(),
        raw=axis_scores.copy(),
    )