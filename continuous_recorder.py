# ============================================================
# continuous_recorder.py
# 会議・会話中の連続計測（フェーズ演技なし）を記録するモジュール
#
# 【重要】フレーム単位で方向スコアを予測→ウィンドウ/セッション単位で
# 平均する設計（mean特徴量への一括predictはしない）。
# これにより生blendshapeの保存は不要（都度破棄）。
# ============================================================
from __future__ import annotations

import uuid
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
AGGREGATED_DIR = DATA_DIR / "aggregated"
CONTINUOUS_SESSIONS_CSV = AGGREGATED_DIR / "continuous_sessions.csv"
CONTINUOUS_WINDOWS_CSV = AGGREGATED_DIR / "continuous_windows.csv"
AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)

DIR_KEYS = ["tension", "anger", "positive", "stable"]


class ContinuousRecorder:
    """
    使い方:
      1. rec = ContinuousRecorder(baseline_id, baseline_frame,
                                    direction_fn=calc_direction_scores,
                                    eq_fn=calc_eq_scores, preset_min=10)
      2. カメラループ内で毎フレーム rec.maybe_record_frame(current_frame) を呼ぶ
         （current_frame が None＝顔未検出の場合は内部でスキップ）
      3. rec.should_auto_stop() が True、または手動停止ボタンで
         rec.finalize() を呼ぶ（集計のみ・CSV保存はまだしない）
      4. UIでnotesを確認・編集後 rec.save(notes) でCSVに保存
    """

    PRESET_CHOICES_MIN = [1, 3, 5, 10, 15, 20, 30]

    def __init__(
        self,
        baseline_id: str,
        baseline_frame: dict,
        direction_fn,             # calc_direction_scores（ML+fallback統合）
        eq_fn,                    # calc_eq_scores
        preset_min: int = 10,
        window_sec: int = 30,
        sample_fps: int = 5,
    ):
        self.session_id = str(uuid.uuid4())
        self.baseline_id = baseline_id
        self.baseline_frame = baseline_frame
        self.direction_fn = direction_fn
        self.eq_fn = eq_fn

        self.preset_sec = preset_min * 60
        self.window_sec = window_sec
        self.sample_interval = 1.0 / sample_fps

        self.started_at = datetime.now().isoformat()
        self.start_time = time.time()
        self.last_sample_time = 0.0

        self.window_index = 0
        self.window_buffer: list[dict] = []     # フレーム毎dir_scores(4値のみ・軽量)
        self.window_results: list[dict] = []    # 確定済みウィンドウ（★空ウィンドウは含まない）
        self.all_frame_scores: list[dict] = []  # セッション全体平均用

        self.skipped_frame_count = 0            # 顔未検出等でスキップした回数
        self.empty_window_count = 0             # ★NEW: 除外した空ウィンドウ数（参考値として保持）

        self._finalized = False
        self.summary: dict | None = None

    # --------------------------------------------------------
    def elapsed_sec(self) -> float:
        return time.time() - self.start_time

    def should_auto_stop(self) -> bool:
        return self.elapsed_sec() >= self.preset_sec

    def progress_ratio(self) -> float:
        return min(1.0, self.elapsed_sec() / self.preset_sec)

    # --------------------------------------------------------
    def maybe_record_frame(self, current_frame: dict | None) -> bool:
        """毎フレーム呼び出し可。内部で5fpsに間引く。"""
        now = time.time()
        if now - self.last_sample_time < self.sample_interval:
            return False
        self.last_sample_time = now

        # 顔未検出（None）はスキップして計測を止めない
        if not current_frame:
            self.skipped_frame_count += 1
            self._maybe_close_windows()
            return False

        # フレーム単位で即予測（学習時の前提と一致させる）
        dir_scores = self.direction_fn(current_frame, self.baseline_frame)
        row = {k: float(getattr(dir_scores, k)) for k in DIR_KEYS}

        self.window_buffer.append(row)
        self.all_frame_scores.append(row)
        # ここでcurrent_frame自体は破棄（保持しない）

        self._maybe_close_windows()
        return True

    # --------------------------------------------------------
    def _maybe_close_windows(self):
        """if→while。処理遅延で複数ウィンドウ分の時間が
        一度に経過した場合でも取りこぼさず確定させる。"""
        while self.elapsed_sec() >= (self.window_index + 1) * self.window_sec:
            self._close_window()

    def _average_scores(self, rows: list[dict]) -> dict:
        if not rows:
            return {k: 0.0 for k in DIR_KEYS}
        df = pd.DataFrame(rows)
        return {k: float(df[k].mean()) for k in DIR_KEYS}

    def _close_window(self):
        if not self.window_buffer:
            # ★FIX: フレームが1件も無いウィンドウ（顔未検出継続等）は
            #        window_results に追加せず「集計対象外」として除外する。
            #        window_indexだけ進めてタイムライン整合性は維持する。
            self.empty_window_count += 1
            self.window_index += 1
            return

        avg_dir = self._average_scores(self.window_buffer)
        eq = self.eq_fn(
            tension=avg_dir["tension"], anger=avg_dir["anger"],
            positive=avg_dir["positive"], stable=avg_dir["stable"],
            source_phase="continuous_window",
        )
        self.window_results.append({
            "window_index": self.window_index,
            "start_sec": round(self.window_index * self.window_sec, 1),
            "end_sec": round((self.window_index + 1) * self.window_sec, 1),
            "frame_count": len(self.window_buffer),
            "dir": avg_dir,
            "eq": eq,
        })
        self.window_index += 1
        self.window_buffer = []

    # --------------------------------------------------------
    def get_progress_scores(self) -> list[dict]:
        """確定済みウィンドウのスコア一覧（途中経過表示用）
        ★FIX後: window_results には空ウィンドウが含まれないため、
        通常このメソッドがNone値を返すことはない（防御的にelse節は残す）"""
        out = []
        for w in self.window_results:
            row = {"window_index": w["window_index"],
                   "start_sec": w["start_sec"], "end_sec": w["end_sec"],
                   "frame_count": w["frame_count"]}
            if w["eq"] is None:
                row.update({f"dir_{k}": None for k in DIR_KEYS})
                row["eq_awareness"] = None
                row["eq_regulation"] = None
                row["eq_social_margin"] = None
            else:
                row.update({f"dir_{k}": round(v, 3) for k, v in w["dir"].items()})
                row["eq_awareness"] = round(w["eq"].awareness_score, 1)
                row["eq_regulation"] = round(w["eq"].regulation_score, 1)
                row["eq_social_margin"] = round(w["eq"].social_margin_score, 1)
            out.append(row)
        return out

    # --------------------------------------------------------
    def finalize(self) -> dict:
        """計測終了時に呼ぶ。集計のみ行い、CSV保存はしない（save()で行う）"""
        if self._finalized:
            return self.summary

        self._close_window()  # 最後の未確定ウィンドウを確定（★空ならwindow_resultsに追加されない）
        overall_dir = self._average_scores(self.all_frame_scores)
        overall_eq = self.eq_fn(
            tension=overall_dir["tension"], anger=overall_dir["anger"],
            positive=overall_dir["positive"], stable=overall_dir["stable"],
            source_phase="continuous_session",
        )

        session_row = {
            "session_id": self.session_id,
            "baseline_id": self.baseline_id,
            "started_at": self.started_at,
            "duration_sec": round(self.elapsed_sec(), 1),
            "frame_count": len(self.all_frame_scores),
            "window_count": len(self.window_results),        # ★有効ウィンドウ数のみ
            "empty_window_count": self.empty_window_count,   # ★NEW: 参考値として保存
            "skipped_frame_count": self.skipped_frame_count,
            "notes": "",  # save()時に上書き
        }
        session_row.update({f"dir_{k}": round(v, 3) for k, v in overall_dir.items()})
        session_row["eq_awareness"] = round(overall_eq.awareness_score, 1)
        session_row["eq_regulation"] = round(overall_eq.regulation_score, 1)
        session_row["eq_social_margin"] = round(overall_eq.social_margin_score, 1)

        self.summary = {"session_row": session_row, "window_rows": self.get_progress_scores()}
        self._finalized = True
        return self.summary

    # --------------------------------------------------------
    def save(self, notes: str = "") -> str:
        """finalize()結果に最終notesを反映してCSVへ保存"""
        if not self._finalized:
            self.finalize()
        self.summary["session_row"]["notes"] = notes

        session_row = self.summary["session_row"]
        window_rows = [{"session_id": self.session_id, **w} for w in self.summary["window_rows"]]

        df_session = pd.DataFrame([session_row])
        if CONTINUOUS_SESSIONS_CSV.exists():
            df_session = pd.concat([pd.read_csv(CONTINUOUS_SESSIONS_CSV), df_session], ignore_index=True)
        df_session.to_csv(CONTINUOUS_SESSIONS_CSV, index=False)

        if window_rows:
            df_windows = pd.DataFrame(window_rows)
            if CONTINUOUS_WINDOWS_CSV.exists():
                df_windows = pd.concat([pd.read_csv(CONTINUOUS_WINDOWS_CSV), df_windows], ignore_index=True)
            df_windows.to_csv(CONTINUOUS_WINDOWS_CSV, index=False)

        print(f"[ContinuousRecorder] 保存完了: {self.session_id} "
              f"duration={session_row['duration_sec']}s "
              f"windows={session_row['window_count']} "
              f"(empty={session_row['empty_window_count']}) "
              f"skipped_frames={session_row['skipped_frame_count']}")
        return self.session_id