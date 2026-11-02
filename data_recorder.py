# ============================================================
# data_recorder.py
# 役割：計測セッションの生データ（52 Blendshape）を保存する
# 保存形式：
#   - フレーム生データ → data/raw/{session_id}_frames.csv
#   - セッション集計   → data/aggregated/sessions.csv
#
# 【baseline_id対応】
#   - sessions.csv に baseline_id 列を追加し、
#     baseline_scores.csv (baseline_store.py管理) と紐付ける
#   - 既存の  には自動移行処理で
#     baseline_id="legacy_unknown" を補完する
# ============================================================
import os
import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── ディレクトリ定義 ──────────────────────────────────────────
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
AGGREGATED_DIR = DATA_DIR / "aggregated"
LABELS_DIR = DATA_DIR / "labels"
SESSIONS_CSV = AGGREGATED_DIR / "sessions.csv"

# 起動時にディレクトリを自動作成
for d in [RAW_DIR, AGGREGATED_DIR, LABELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# DataRecorder クラス
# 1セッション分のデータを収集・保存する責務を持つ
# ============================================================
class DataRecorder:
    def __init__(self):
        # セッションIDは UUID4 で一意に生成
        self.session_id = str(uuid.uuid4())
        self.started_at = datetime.now().isoformat()
        self.frames: list[dict] = []  # フレームデータの一時バッファ
        self.frame_index = 0

    # ----------------------------------------------------------
    # record_frame()
    # 1フレーム分の Blendshape データをバッファに追加する
    # 引数:
    #   blendshapes : dict {name: value, ...} 52個
    #   phase       : str  現在の計測フェーズ名
    # ----------------------------------------------------------
    def record_frame(self, blendshapes: dict, phase: str):
        row = {
            "frame_index": self.frame_index,
            "timestamp_ms": int(datetime.now().timestamp() * 1000),
            "phase": phase,
        }
        # 52 Blendshape を個別カラムとして追加
        row.update(blendshapes)
        self.frames.append(row)
        self.frame_index += 1

    # ----------------------------------------------------------
    # save_session()
    # セッション終了時に呼び出す
    #   ① フレーム生データを CSV で保存
    #   ② セッション集計行を sessions.csv に追記
    # 引数:
    #   direction_scores : dict {tension, anger, positive, stable}
    #   eq_scores        : dict {awareness, regulation, social_margin}
    #   baseline_id      : str  このセッションが使用したベースラインのID
    #   notes            : str  任意メモ（省略可）
    # ----------------------------------------------------------
    def save_session(
        self,
        direction_scores: dict,
        eq_scores: dict,
        baseline_id: str | None = None,
        notes: str = ""
    ) -> str:
        """保存した session_id を返す"""
        if not self.frames:
            print("[DataRecorder] フレームデータが空のため保存をスキップ")
            return self.session_id

        # ── ① フレーム生データを CSV 保存 ──────────────────────
        df_frames = pd.DataFrame(self.frames)
        csv_path = RAW_DIR / f"{self.session_id}_frames.csv"
        df_frames.to_csv(csv_path, index=False)
        print(f"[DataRecorder] 生データ保存: {csv_path} ({len(df_frames)} frames)")

        # ── ② フェーズ別統計をセッション行として集計 ──────────
        session_row = self._build_session_row(
            df_frames, direction_scores, eq_scores, baseline_id, notes
        )

        # sessions.csv に追記（なければ新規作成）
        df_session = pd.DataFrame([session_row])
        if SESSIONS_CSV.exists():
            df_existing = pd.read_csv(SESSIONS_CSV)
            df_session = pd.concat([df_existing, df_session], ignore_index=True)
        df_session.to_csv(SESSIONS_CSV, index=False)
        print(f"[DataRecorder] セッション集計保存: {SESSIONS_CSV}")

        return self.session_id

    # ----------------------------------------------------------
    # _build_session_row() ← 内部メソッド
    # sessions.csv に書き込む1行分の辞書を構築する
    # ----------------------------------------------------------
    def _build_session_row(
        self,
        df_frames: pd.DataFrame,
        direction_scores: dict,
        eq_scores: dict,
        baseline_id: str | None,
        notes: str
    ) -> dict:
        row = {
            "session_id": self.session_id,
            "baseline_id": baseline_id if baseline_id else "legacy_unknown",
            "started_at": self.started_at,
            "frame_count": len(df_frames),
            "notes": notes,
        }

        # 方向スコア
        for k, v in direction_scores.items():
            row[f"dir_{k}"] = round(v, 3)

        # EQ スコア
        for k, v in eq_scores.items():
            row[f"eq_{k}"] = round(v, 3)

        # フェーズ別 Blendshape 平均・標準偏差
        bs_cols = [c for c in df_frames.columns
                   if c not in ("frame_index", "timestamp_ms", "phase")]
        for phase in df_frames["phase"].unique():
            df_p = df_frames[df_frames["phase"] == phase][bs_cols]
            for col in bs_cols:
                row[f"{phase}_mean_{col}"] = round(float(df_p[col].mean()), 5)
                row[f"{phase}_std_{col}"] = round(float(df_p[col].std()), 5)

        return row

    # ----------------------------------------------------------
    # get_neutral_summary()
    # neutral フェーズの Blendshape 平均を返す（EQ計算用）
    # ----------------------------------------------------------
    def get_neutral_summary(self) -> dict | None:
        if not self.frames:
            return None
        df = pd.DataFrame(self.frames)
        df_n = df[df["phase"] == "neutral"]
        if df_n.empty:
            return None
        bs_cols = [c for c in df.columns
                   if c not in ("frame_index", "timestamp_ms", "phase")]
        return df_n[bs_cols].mean().to_dict()


# ============================================================
# 既存データ移行処理
# 過去に baseline_id 列なしで保存された sessions.csv がある場合、
# 列を追加して "legacy_unknown" を補完する（モジュール読込時に一度だけ実行）
# ============================================================
def _migrate_sessions_csv_add_baseline_id():
    if not SESSIONS_CSV.exists():
        return
    df = pd.read_csv(SESSIONS_CSV)
    if "baseline_id" in df.columns:
        return
    df.insert(1, "baseline_id", "legacy_unknown")
    df.to_csv(SESSIONS_CSV, index=False)
    print("[DataRecorder] sessions.csv に baseline_id 列を移行しました")


_migrate_sessions_csv_add_baseline_id()