# ============================================================
# train_phase_classifier_delta.py
# 概要: 各フレームの生blendshape値からセッション紐づけのbaseline値を
#       引き算した"delta特徴量"でphase(4クラス)分類モデルを学習する。
#
# train_phase_classifier.py との違い:
# - 特徴量が「絶対値」→「baselineからの変化量」になるだけで、
#   Lasso→RFのパイプラインは完全に共通。
# - baseline_id が legacy_unknown / 未登録のセッションは学習対象外
#   （理由をログに明示してスキップする）。
# ============================================================
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from data_recorder import RAW_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGG_DIR = os.path.join(BASE_DIR, "data", "aggregated")

SESSIONS_CSV = os.path.join(AGG_DIR, "sessions.csv")
BASELINE_CSV = os.path.join(AGG_DIR, "baseline_scores.csv")

META_COLS = ["frame_index", "timestamp_ms", "phase", "session_id", "baseline_id"]
MODEL_OUTPUT_PATH = "phase_classifier_rf_delta.pkl"


def load_baseline_lookup(path=BASELINE_CSV):
    """baseline_id -> 51blendshapeのSeries、を作る辞書"""
    df_b = pd.read_csv(path)
    meta = ["baseline_id", "date", "recorded_at", "is_manual"]
    feature_cols = [c for c in df_b.columns if c not in meta]
    # 重複baseline_idがあれば最後の行を優先(要注意ログ)
    dup = df_b["baseline_id"].duplicated().sum()
    if dup:
        print(f"⚠️ baseline_scores.csvに重複baseline_idが{dup}件あります")
    lookup = {
        row["baseline_id"]: row[feature_cols].astype(float)
        for _, row in df_b.iterrows()
    }
    return lookup, feature_cols


def load_session_baseline_map(path=SESSIONS_CSV):
    df_s = pd.read_csv(path)
    return dict(zip(df_s["session_id"], df_s["baseline_id"]))


def load_all_frames_with_delta(raw_dir=RAW_DIR) -> pd.DataFrame:
    baseline_lookup, baseline_feature_cols = load_baseline_lookup()
    session_baseline_map = load_session_baseline_map()

    pattern = os.path.join(str(raw_dir), "*_frames.csv")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"フレームCSVが見つかりません: {pattern}")

    frames, skipped = [], []
    for path in paths:
        session_id = os.path.basename(path).replace("_frames.csv", "")
        baseline_id = session_baseline_map.get(session_id)

        if baseline_id is None or baseline_id == "legacy_unknown":
            skipped.append((session_id, "no_baseline_linked")); continue
        if baseline_id not in baseline_lookup:
            skipped.append((session_id, f"baseline_missing:{baseline_id}")); continue

        df = pd.read_csv(path)
        missing = [c for c in baseline_feature_cols if c not in df.columns]
        if missing:
            skipped.append((session_id, f"feature_mismatch:{missing[:3]}...")); continue

        baseline_vec = baseline_lookup[baseline_id]
        df[baseline_feature_cols] = df[baseline_feature_cols] - baseline_vec  # ★deltaに変換
        df["session_id"] = session_id
        df["baseline_id"] = baseline_id
        frames.append(df)

    if skipped:
        print(f"[load_all_frames_with_delta] スキップ: {len(skipped)}件")
        for sid, reason in skipped:
            print(f"  - {sid}: {reason}")

    if not frames:
        raise RuntimeError(
            "baselineと紐づく有効なセッションが0件です。"
            "sessions.csv / baseline_scores.csv を確認してください。"
        )

    df_all = pd.concat(frames, ignore_index=True)
    print(f"[load_all_frames_with_delta] {len(frames)}セッション / {len(df_all)}フレーム 読込")
    print(f"[load_all_frames_with_delta] phase内訳:\n{df_all['phase'].value_counts()}")
    return df_all


def main():
    df = load_all_frames_with_delta()
    feature_cols = [c for c in df.columns if c not in META_COLS]
    assert len(feature_cols) == 51, f"想定外の特徴量数: {len(feature_cols)}"

    # ★ PyArrow拡張配列対策: 明示的にnumpyのネイティブ配列へ変換
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["phase"].astype(str).to_numpy()
    groups = df["session_id"].astype(str).to_numpy()

    n_sessions = len(np.unique(groups))
    print(f"[main] baseline紐づけ済セッション数: {n_sessions}")
    if n_sessions < 5:
        print("⚠️ セッション数が少なくGroupKFoldの信頼性が低い可能性があります")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_splits = min(5, n_sessions)
    gkf = GroupKFold(n_splits=n_splits)

    # Step1: Lasso特徴量選択（既存と同一ロジック）
    cv_lasso = list(gkf.split(X_scaled, y, groups))
    lasso_cv = LogisticRegressionCV(
        Cs=10, cv=cv_lasso, penalty="l1", solver="saga", max_iter=5000, n_jobs=-1,
    )
    lasso_cv.fit(X_scaled, y)
    coef_abs = np.abs(lasso_cv.coef_).sum(axis=0)
    selected_idx = np.where(coef_abs > 1e-4)[0]
    selected_features = [feature_cols[i] for i in selected_idx]
    print(f"[main] 選択された特徴量(delta): {len(selected_features)} / {len(feature_cols)}")
    X_selected = X_scaled[:, selected_idx]

    # Step2: RandomForest + GridSearch（既存と同一ロジック）
    param_grid = {"n_estimators": [200, 400, 600], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 3, 5]}
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    cv_grid = list(gkf.split(X_selected, y, groups))
    grid = GridSearchCV(rf, param_grid, cv=cv_grid, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_selected, y)
    best_rf = grid.best_estimator_
    print(f"[main] Best params: {grid.best_params_} / f1_macro: {grid.best_score_:.4f}")

    oof_pred = cross_val_predict(best_rf, X_selected, y, cv=list(gkf.split(X_selected, y, groups)), n_jobs=-1)
    print("\n=== Classification Report (delta特徴量) ===")
    print(classification_report(y, oof_pred))

    labels = [l for l in ["smile", "brow_up", "angry", "neutral"] if l in np.unique(y)]
    print(pd.DataFrame(confusion_matrix(y, oof_pred, labels=labels), index=labels, columns=labels))

    importances = pd.Series(best_rf.feature_importances_, index=selected_features).sort_values(ascending=False)
    print("\n=== 重要度Top15(delta) ===")
    print(importances.head(15))

    joblib.dump(
        {"model": best_rf, "scaler": scaler, "selected_idx": selected_idx,
         "feature_cols": feature_cols, "selected_features": selected_features,
         "mode": "delta_from_baseline"},
        MODEL_OUTPUT_PATH,
    )
    print(f"\n[main] 保存完了: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()