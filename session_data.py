# ============================================================
# session_data.py
# セッション中の時系列データ管理・保存クラス
#
# 保存データ構造:
#   - フレームインデックス
#   - タイムスタンプ (ms)
#   - 52 Blendshape差分値 (bs_xxx)
#   - 3方向感情スコア (tension, anger, positive)
#   - 移動平均スコア (smoothed_xxx)
#
# 保存形式:
#   - CSV: ML学習・分析用
#   - JSON: サマリー統計（セッション概要）
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ============================================================
# SessionDataManager
# ============================================================

class SessionDataManager:
    """
    1セッション分の時系列データを管理するクラス

    Parameters
    ----------
    session_id : str, optional
        セッションID（指定しない場合は日時から自動生成）
    save_dir : str
        データ保存ディレクトリ（デフォルト: "./sessions"）
    max_records : int
        メモリ上に保持する最大レコード数（デフォルト: 10000）
        超過した場合は古いレコードから削除

    使用例
    ------
    manager = SessionDataManager()
    manager.append(blendshape_delta, emotion_scores, frame_index=0)
    csv_path, json_path = manager.save()
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        save_dir: str = "./sessions",
        max_records: int = 10000,
    ):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = save_dir
        self.max_records = max_records

        # レコードリスト（辞書のリスト）
        self.records: List[Dict] = []

        # セッション開始時刻
        self.start_time = datetime.now()

        # 保存済みパス
        self._saved_csv_path: Optional[str] = None
        self._saved_json_path: Optional[str] = None

    # ──────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ──────────────────────────────────────────────────────

    def append(
        self,
        blendshape_delta: Dict[str, float],
        emotion_scores: Dict[str, float],
        frame_index: int,
    ):
        """
        1フレーム分のデータをレコードに追加する

        Parameters
        ----------
        blendshape_delta : dict
            {blendshape名: 差分値} の辞書
        emotion_scores : dict
            EmotionScoreCalculator.calculate()の戻り値
        frame_index : int
            フレーム番号（0始まり）
        """
        # 経過時間 (ms) を計算
        elapsed_ms = (
            datetime.now() - self.start_time
        ).total_seconds() * 1000.0

        # レコード構築
        record: Dict = {
            "session_id":   self.session_id,
            "frame_index":  frame_index,
            "timestamp_ms": round(elapsed_ms, 1),
        }

        # Blendshape差分値を "bs_" プレフィックスで追加
        for bs_name, value in blendshape_delta.items():
            record[f"bs_{bs_name}"] = round(float(value), 6)

        # 感情スコアを追加
        for score_name, value in emotion_scores.items():
            record[score_name] = round(float(value), 4)

        # レコード追加
        self.records.append(record)

        # 最大レコード数を超えた場合は先頭から削除
        if len(self.records) > self.max_records:
            self.records.pop(0)

    def to_dataframe(self) -> pd.DataFrame:
        """
        記録済みデータをpandas DataFrameに変換する

        Returns
        -------
        pd.DataFrame
            全レコードを含むDataFrame
            レコードがない場合は空のDataFrameを返す
        """
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)

    def save(self) -> Tuple[str, str]:
        """
        セッションデータをCSVとJSONで保存する

        Returns
        -------
        tuple[str, str]
            (csv_path, json_path) 保存先パスのタプル

        Raises
        ------
        ValueError
            レコードが空の場合
        """
        if not self.records:
            raise ValueError("保存するレコードがありません。")

        os.makedirs(self.save_dir, exist_ok=True)

        df = self.to_dataframe()

        # ── CSV保存（ML学習用） ──
        csv_path = os.path.join(
            self.save_dir, f"{self.session_id}.csv"
        )
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # ── サマリーJSON保存 ──
        summary = self._compute_summary(df)
        json_path = os.path.join(
            self.save_dir, f"{self.session_id}_summary.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self._saved_csv_path = csv_path
        self._saved_json_path = json_path

        return csv_path, json_path

    def get_recent_scores(
        self,
        n_frames: int = 30,
    ) -> Dict[str, List[float]]:
        """
        直近Nフレームの感情スコアを方向ごとのリストで返す

        Parameters
        ----------
        n_frames : int
            取得するフレーム数

        Returns
        -------
        dict
            {
              "tension":  [float, ...],
              "anger":    [float, ...],
              "positive": [float, ...],
            }
        """
        recent = self.records[-n_frames:] if self.records else []
        result: Dict[str, List[float]] = {
            "tension":  [],
            "anger":    [],
            "positive": [],
        }
        for record in recent:
            for direction in result.keys():
                result[direction].append(
                    record.get(f"smoothed_{direction}", 50.0)
                )
        return result

    def get_session_stats(self) -> Dict:
        """
        現在のセッション統計を返す（保存前でも利用可能）

        Returns
        -------
        dict
            セッション統計情報
        """
        if not self.records:
            return {"status": "no_data"}

        df = self.to_dataframe()
        return self._compute_summary(df)

    def clear(self):
        """
        レコードをクリアして新しいセッションを開始する
        """
        self.records = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self._saved_csv_path = None
        self._saved_json_path = None

    @property
    def record_count(self) -> int:
        """現在のレコード数"""
        return len(self.records)

    @property
    def duration_sec(self) -> float:
        """セッション経過時間（秒）"""
        return (datetime.now() - self.start_time).total_seconds()

    # ──────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ──────────────────────────────────────────────────────

    def _compute_summary(self, df: pd.DataFrame) -> Dict:
        """
        セッションサマリー統計を計算する

        Parameters
        ----------
        df : pd.DataFrame
            to_dataframe()の出力

        Returns
        -------
        dict
            サマリー統計辞書
        """
        score_directions = ["tension", "anger", "positive"]
        smoothed_directions = [f"smoothed_{d}" for d in score_directions]

        summary = {
            "session_id":    self.session_id,
            "start_time":    self.start_time.isoformat(),
            "total_frames":  len(df),
            "duration_sec":  round(
                df["timestamp_ms"].max() / 1000.0
                if "timestamp_ms" in df.columns and len(df) > 0
                else 0.0,
                2,
            ),
            "raw_scores":      {},
            "smoothed_scores": {},
        }

        # 生スコア統計
        for col in score_directions:
            if col in df.columns:
                summary["raw_scores"][col] = self._col_stats(df[col])

        # 移動平均スコア統計
        for col in smoothed_directions:
            if col in df.columns:
                key = col.replace("smoothed_", "")
                summary["smoothed_scores"][key] = self._col_stats(df[col])

        # 支配的方向（平均スコアが最高の方向）
        if summary["smoothed_scores"]:
            dominant = max(
                summary["smoothed_scores"],
                key=lambda d: summary["smoothed_scores"][d].get("mean", 0),
            )
            summary["dominant_direction"] = dominant

        return summary

    @staticmethod
    def _col_stats(series: pd.Series) -> Dict:
        """
        数値列の基本統計を計算する

        Returns
        -------
        dict
            mean, std, max, min, median
        """
        return {
            "mean":   round(float(series.mean()),   2),
            "std":    round(float(series.std()),    2),
            "max":    round(float(series.max()),    2),
            "min":    round(float(series.min()),    2),
            "median": round(float(series.median()), 2),
        }


# ============================================================
# SessionHistoryLoader（過去セッション読み込み）
# ============================================================

class SessionHistoryLoader:
    """
    保存済みセッションデータを読み込むクラス

    使用例
    ------
    loader = SessionHistoryLoader(save_dir="./sessions")
    sessions = loader.list_sessions()
    df = loader.load_session("20240101_120000")
    """

    def __init__(self, save_dir: str = "./sessions"):
        self.save_dir = save_dir

    def list_sessions(self) -> List[Dict]:
        """
        保存済みセッション一覧を返す

        Returns
        -------
        list[dict]
            [{session_id, csv_path, json_path, ...}, ...]
            新しい順にソート
        """
        if not os.path.exists(self.save_dir):
            return []

        sessions = []
        for fname in os.listdir(self.save_dir):
            if fname.endswith("_summary.json"):
                session_id = fname.replace("_summary.json", "")
                json_path = os.path.join(self.save_dir, fname)
                csv_path = os.path.join(self.save_dir, f"{session_id}.csv")

                try:
                    with open(json_path, encoding="utf-8") as f:
                        summary = json.load(f)
                    sessions.append({
                        "session_id": session_id,
                        "csv_path":   csv_path,
                        "json_path":  json_path,
                        "summary":    summary,
                    })
                except Exception:
                    continue

        # 新しい順にソート
        sessions.sort(key=lambda x: x["session_id"], reverse=True)
        return sessions

    def load_session(self, session_id: str) -> Optional[pd.DataFrame]:
        """
        指定セッションのCSVデータを読み込む

        Parameters
        ----------
        session_id : str
            セッションID

        Returns
        -------
        pd.DataFrame or None
            読み込んだDataFrame、失敗時はNone
        """
        csv_path = os.path.join(self.save_dir, f"{session_id}.csv")
        if not os.path.exists(csv_path):
            return None
        try:
            return pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            return None

    def load_summary(self, session_id: str) -> Optional[Dict]:
        """
        指定セッションのサマリーJSONを読み込む
        """
        json_path = os.path.join(
            self.save_dir, f"{session_id}_summary.json"
        )
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


# ============================================================
# 動作確認
# ============================================================

if __name__ == "__main__":
    print("=== SessionDataManager テスト ===\n")

    manager = SessionDataManager(
        session_id="test_session",
        save_dir="./sessions_test",
    )

    # ダミーデータを10フレーム追加
    for i in range(10):
        delta = {
            "mouthSmileLeft":  0.1 * i,
            "mouthSmileRight": 0.1 * i,
            "browDownLeft":    0.05 * i,
        }
        scores = {
            "tension":           40.0 + i,
            "anger":             30.0 + i,
            "positive":          50.0 + i,
            "smoothed_tension":  40.0 + i * 0.5,
            "smoothed_anger":    30.0 + i * 0.5,
            "smoothed_positive": 50.0 + i * 0.5,
        }
        manager.append(delta, scores, frame_index=i)

    print(f"レコード数: {manager.record_count}")
    print(f"経過時間: {manager.duration_sec:.2f}秒")

    # 統計確認
    stats = manager.get_session_stats()
    print(f"\nサマリー: {json.dumps(stats, indent=2, ensure_ascii=False)}")

    # 保存テスト
    csv_path, json_path = manager.save()
    print(f"\nCSV保存先: {csv_path}")
    print(f"JSON保存先: {json_path}")