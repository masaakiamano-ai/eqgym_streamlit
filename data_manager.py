# ============================================================
# data_manager.py
# FR-08: セッションデータの保存・読み込み管理
# ============================================================

import json
import csv
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd

# ── 保存ディレクトリ定義 ──────────────────────────────────────
DATA_DIR     = "data"
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
SUMMARY_CSV  = os.path.join(DATA_DIR, "summary.csv")

# CSV のカラム定義
CSV_COLUMNS = [
    "session_id",
    "timestamp",
    "tension_ratio",
    "anger_ratio",
    "positive_ratio",
    "stable_ratio",
    "awareness_score",
    "regulation_score",
    "social_margin_score",
    "reflection_score",
    "memo",
]


# ── セッションデータ構造 ──────────────────────────────────────
@dataclass
class SessionData:
    """1回の計測セッションを表すデータクラス"""

    session_id: str           # 例: 20240601_143022
    timestamp: str            # ISO形式: 2024-06-01T14:30:22
    
    # 4方向比率（neutral フェーズ平均）
    tension_ratio:  float
    anger_ratio:    float
    positive_ratio: float
    stable_ratio:   float
    
    # EQスコア（0〜100）
    awareness_score:      float
    regulation_score:     float
    social_margin_score:  float
    reflection_score:     float
    
    # フェーズ別生データ（オプション）
    raw_phase_data: Optional[dict] = None
    
    # メモ（ユーザー入力）
    memo: str = ""


def ensure_dirs():
    """保存用ディレクトリを作成する"""
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def generate_session_id() -> str:
    """現在時刻からセッションIDを生成する"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── 保存処理 ─────────────────────────────────────────────────
def save_session(session: SessionData) -> str:
    """
    セッションデータを保存する
    - JSON: 生データ含む詳細データ
    - CSV : サマリー（追記）
    戻り値: 保存したJSONファイルパス
    """
    ensure_dirs()

    # ── JSON 保存 ──────────────────────────────────────────
    json_path = os.path.join(
        SESSIONS_DIR, f"{session.session_id}_session.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(session), f, ensure_ascii=False, indent=2)

    # ── CSV 追記 ───────────────────────────────────────────
    file_exists = os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        # ファイルが新規の場合はヘッダーを書く
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "session_id":           session.session_id,
            "timestamp":            session.timestamp,
            "tension_ratio":        round(session.tension_ratio,  3),
            "anger_ratio":          round(session.anger_ratio,    3),
            "positive_ratio":       round(session.positive_ratio, 3),
            "stable_ratio":         round(session.stable_ratio,   3),
            "awareness_score":      round(session.awareness_score,     1),
            "regulation_score":     round(session.regulation_score,    1),
            "social_margin_score":  round(session.social_margin_score, 1),
            "reflection_score":     round(session.reflection_score,    1),
            "memo":                 session.memo,
        })

    return json_path


# ── 読み込み処理 ──────────────────────────────────────────────
def load_summary() -> pd.DataFrame:
    """
    summary.csv を DataFrame として読み込む
    ファイルが存在しない場合は空の DataFrame を返す
    """
    if not os.path.exists(SUMMARY_CSV):
        return pd.DataFrame(columns=CSV_COLUMNS)
    
    df = pd.read_csv(SUMMARY_CSV, encoding="utf-8")
    # timestamp を datetime 型に変換
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_session_json(session_id: str) -> Optional[dict]:
    """
    指定セッションIDの JSON を読み込む
    存在しない場合は None を返す
    """
    json_path = os.path.join(SESSIONS_DIR, f"{session_id}_session.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_session(session_id: str) -> bool:
    """
    指定セッションIDのJSONを削除し、CSVからも除外する
    戻り値: 削除成功なら True
    """
    # JSON 削除
    json_path = os.path.join(SESSIONS_DIR, f"{session_id}_session.json")
    if os.path.exists(json_path):
        os.remove(json_path)

    # CSV から該当行を除外して上書き
    if os.path.exists(SUMMARY_CSV):
        df = load_summary()
        df = df[df["session_id"] != session_id]
        df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")

    return True


# ── SessionData 生成ヘルパー ──────────────────────────────────
def build_session(
    ratios: dict,
    eq_scores,
    raw_phase_data: Optional[dict] = None,
    memo: str = "",
) -> SessionData:
    """
    ratios と EQScores オブジェクトから SessionData を生成する

    Parameters
    ----------
    ratios : dict
        {"tension": float, "anger": float,
         "positive": float, "stable": float}
    eq_scores : EQScores
        eq_score.py の EQScores dataclass
    raw_phase_data : dict, optional
        フェーズ別 Blendshape 生データ
    memo : str
        ユーザーメモ
    """
    sid = generate_session_id()
    return SessionData(
        session_id           = sid,
        timestamp            = datetime.now().isoformat(),
        tension_ratio        = ratios.get("tension",  0.0),
        anger_ratio          = ratios.get("anger",    0.0),
        positive_ratio       = ratios.get("positive", 0.0),
        stable_ratio         = ratios.get("stable",   0.0),
        awareness_score      = eq_scores.awareness_score,
        regulation_score     = eq_scores.regulation_score,
        social_margin_score  = eq_scores.social_margin_score,
        reflection_score     = eq_scores.reflection_score,
        raw_phase_data       = raw_phase_data,
        memo                 = memo,
    )