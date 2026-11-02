# ============================================================
# calibration.py
# 概要: 収集済みCSVデータからニュートラルベースラインと
#       各フェーズの統計量を計算・保存・読み込みするモジュール
#
# 処理フロー:
#   CSVロード → フェーズ別分離 → 統計計算 → JSON保存/読み込み
#
# 出力ファイル: calibration_data.json
# ============================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# 定数定義
# ============================================================

CALIBRATION_FILE = "calibration_data.json"

# CSVの52特徴量カラム名（frame_index, timestamp_ms, phase を除く）
BLENDSHAPE_KEYS = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

# フェーズ名定義
PHASE_NEUTRAL  = "neutral"
PHASE_SMILE    = "smile"
PHASE_BROW_UP  = "brow_up"
PHASE_ANGRY    = "angry"

ALL_PHASES = [PHASE_NEUTRAL, PHASE_SMILE, PHASE_BROW_UP, PHASE_ANGRY]

# ============================================================
# データクラス
# ============================================================

@dataclass
class PhaseStats:
    """
    1フェーズ分の統計量を保持するデータクラス

    Attributes:
        phase   : フェーズ名
        mean    : 各特徴量の平均値 {key: float}
        std     : 各特徴量の標準偏差 {key: float}
        n_frames: フレーム数
    """
    phase   : str
    mean    : dict = field(default_factory=dict)
    std     : dict = field(default_factory=dict)
    n_frames: int  = 0


@dataclass
class CalibrationData:
    """
    全フェーズの統計量とNORMALIZE_MAX推定値を保持するクラス

    Attributes:
        phases        : フェーズ名 → PhaseStats の辞書
        normalize_max : 方向別の正規化上限値推定 {direction: float}
        baseline      : ニュートラルフェーズの平均値 {key: float}
        n_sessions    : 収集セッション数
    """
    phases        : dict = field(default_factory=dict)
    normalize_max : dict = field(default_factory=dict)
    baseline      : dict = field(default_factory=dict)
    n_sessions    : int  = 0


# ============================================================
# CalibrationBuilder クラス
# ============================================================

class CalibrationBuilder:
    """
    CSVファイルからキャリブレーションデータを構築するクラス

    使用方法:
        builder = CalibrationBuilder()
        builder.load_csv("frames.csv")
        builder.load_csv("frames2.csv")  # 複数セッション追加可能
        cal_data = builder.build()
        builder.save(cal_data)
    """

    def __init__(self):
        """初期化: フェーズ別データフレームリストを準備"""
        # フェーズ別に生データを蓄積するリスト
        self._phase_dfs: dict[str, list[pd.DataFrame]] = {
            phase: [] for phase in ALL_PHASES
        }
        self._n_sessions = 0

    # ----------------------------------------------------------
    # CSVロード
    # ----------------------------------------------------------

    def load_csv(self, csv_path: str) -> None:
        """
        CSVファイルを読み込み、フェーズ別に分離して蓄積する

        Args:
            csv_path: CSVファイルパス
        """
        path = Path(csv_path)
        if not path.exists():
            print(f"[CalibrationBuilder] ファイルが見つかりません: {csv_path}")
            return

        df = pd.read_csv(csv_path)

        # 必須カラム確認
        required = {"phase"} | set(BLENDSHAPE_KEYS)
        missing = required - set(df.columns)
        if missing:
            print(f"[CalibrationBuilder] 不足カラム: {missing}")
            return

        # フェーズ別に分離して蓄積
        for phase in ALL_PHASES:
            phase_df = df[df["phase"] == phase][BLENDSHAPE_KEYS].copy()
            if len(phase_df) > 0:
                self._phase_dfs[phase].append(phase_df)

        self._n_sessions += 1
        print(f"[CalibrationBuilder] ロード完了: {csv_path} "
              f"(セッション {self._n_sessions})")

    # ----------------------------------------------------------
    # キャリブレーションデータ構築
    # ----------------------------------------------------------

    def build(self) -> CalibrationData:
        """
        蓄積データからキャリブレーションデータを構築する

        Returns:
            CalibrationData: 統計量とNORMALIZE_MAX推定値
        """
        cal = CalibrationData(n_sessions=self._n_sessions)

        # ── フェーズ別統計計算 ────────────────────────────────
        for phase in ALL_PHASES:
            dfs = self._phase_dfs[phase]
            if not dfs:
                print(f"[CalibrationBuilder] データなし: {phase}")
                continue

            # 全セッションを結合
            combined = pd.concat(dfs, ignore_index=True)

            stats = PhaseStats(
                phase    = phase,
                mean     = combined.mean().to_dict(),
                std      = combined.std().to_dict(),
                n_frames = len(combined),
            )
            cal.phases[phase] = stats
            print(f"[CalibrationBuilder] {phase}: {len(combined)}フレーム")

        # ── ベースライン設定（neutralフェーズの平均） ──────────
        if PHASE_NEUTRAL in cal.phases:
            cal.baseline = cal.phases[PHASE_NEUTRAL].mean.copy()
        else:
            # neutralがない場合は全特徴量0.0
            cal.baseline = {k: 0.0 for k in BLENDSHAPE_KEYS}
            print("[CalibrationBuilder] WARNING: neutral フェーズなし → baseline=0")

        # ── NORMALIZE_MAX 推定 ────────────────────────────────
        cal.normalize_max = self._estimate_normalize_max(cal)

        return cal

    # ----------------------------------------------------------
    # NORMALIZE_MAX 推定
    # ----------------------------------------------------------

    def _estimate_normalize_max(self, cal: CalibrationData) -> dict:
        """
        フェーズ別平均値からNORMALIZE_MAXを推定する

        推定ロジック:
        - tension_max  = brow_up フェーズのTENSION_KEYS平均デルタ * 1.2
        - anger_max    = angry フェーズのANGER_KEYS平均デルタ * 1.2
        - positive_max = smile フェーズのPOSITIVE_KEYS平均デルタ * 1.2
        - overall_max  = neutral フェーズの全特徴量変化量 * 1.5

        Args:
            cal: 構築中のCalibrationData

        Returns:
            dict: {"tension": float, "anger": float,
                   "positive": float, "overall": float}
        """
        baseline = cal.baseline

        # ── tension_max: brow_up フェーズから推定 ──────────────
        TENSION_KEYS = [
            "browInnerUp", "eyeWideLeft", "eyeWideRight", "jawOpen"
        ]
        tension_max = self._calc_delta_mean(
            cal, PHASE_BROW_UP, TENSION_KEYS, baseline
        ) * 1.2

        # ── anger_max: angry フェーズから推定 ──────────────────
        # 改善②反映: noseSneer, mouthFrown を追加
        ANGER_KEYS = [
            "browDownLeft", "browDownRight",
            "jawForward",
            "mouthPressLeft", "mouthPressRight",
            "noseSneerLeft", "noseSneerRight",   # 追加
            "mouthFrownLeft", "mouthFrownRight", # 追加
        ]
        anger_max = self._calc_delta_mean(
            cal, PHASE_ANGRY, ANGER_KEYS, baseline
        ) * 1.2

        # ── positive_max: smile フェーズから推定 ────────────────
        POSITIVE_KEYS = [
            "mouthSmileLeft", "mouthSmileRight",
            "cheekSquintLeft", "cheekSquintRight",
            "eyeSquintLeft", "eyeSquintRight",
        ]
        positive_max = self._calc_delta_mean(
            cal, PHASE_SMILE, POSITIVE_KEYS, baseline
        ) * 1.2

        # ── overall_max: neutral フェーズの微小変化から推定 ──────
        # neutral フェーズ内の標準偏差の平均 * 3.0
        overall_max = 0.06  # デフォルト値
        if PHASE_NEUTRAL in cal.phases:
            std_vals = list(cal.phases[PHASE_NEUTRAL].std.values())
            if std_vals:
                overall_max = float(np.mean(std_vals)) * 3.0
                overall_max = max(0.03, min(0.10, overall_max))  # 範囲制限

        # ── 下限ガード（ゼロ除算防止） ────────────────────────
        tension_max  = max(0.05, tension_max)
        anger_max    = max(0.03, anger_max)
        positive_max = max(0.05, positive_max)

        result = {
            "tension" : round(tension_max,  4),
            "anger"   : round(anger_max,    4),
            "positive": round(positive_max, 4),
            "overall" : round(overall_max,  4),
        }

        print(f"[CalibrationBuilder] NORMALIZE_MAX推定値: {result}")
        return result

    def _calc_delta_mean(
        self,
        cal    : CalibrationData,
        phase  : str,
        keys   : list,
        baseline: dict,
    ) -> float:
        """
        指定フェーズの指定キーにおけるデルタ平均を計算する

        Args:
            cal     : CalibrationData
            phase   : フェーズ名
            keys    : 対象キーリスト
            baseline: ベースライン平均値辞書

        Returns:
            float: デルタ平均値
        """
        if phase not in cal.phases:
            return 0.10  # フォールバック値

        phase_mean = cal.phases[phase].mean
        deltas = []
        for k in keys:
            if k in phase_mean and k in baseline:
                delta = max(0.0, phase_mean[k] - baseline[k])
                deltas.append(delta)

        if not deltas:
            return 0.10

        return float(np.mean(deltas))

    # ----------------------------------------------------------
    # 保存・読み込み
    # ----------------------------------------------------------

    def save(self, cal: CalibrationData, path: str = CALIBRATION_FILE) -> None:
        """
        キャリブレーションデータをJSONに保存する

        Args:
            cal : CalibrationData
            path: 保存先ファイルパス
        """
        data = {
            "n_sessions"   : cal.n_sessions,
            "baseline"     : cal.baseline,
            "normalize_max": cal.normalize_max,
            "phases"       : {},
        }

        for phase, stats in cal.phases.items():
            data["phases"][phase] = {
                "mean"    : stats.mean,
                "std"     : stats.std,
                "n_frames": stats.n_frames,
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[CalibrationBuilder] 保存完了: {path}")


# ============================================================
# CalibrationLoader クラス
# ============================================================

class CalibrationLoader:
    """
    保存済みキャリブレーションデータを読み込むクラス

    使用方法:
        loader = CalibrationLoader()
        cal = loader.load()
        baseline = loader.get_baseline_frame()
        norm_max = loader.get_normalize_max()
    """

    def __init__(self, path: str = CALIBRATION_FILE):
        """
        Args:
            path: JSONファイルパス
        """
        self._path = path
        self._data: Optional[dict] = None

    def load(self) -> bool:
        """
        JSONからキャリブレーションデータを読み込む

        Returns:
            bool: 読み込み成功なら True
        """
        p = Path(self._path)
        if not p.exists():
            print(f"[CalibrationLoader] ファイルなし: {self._path}")
            return False

        with open(self._path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        print(f"[CalibrationLoader] 読み込み完了: {self._path} "
              f"(セッション数: {self._data.get('n_sessions', '?')})")
        return True

    def get_baseline_frame(self):
        """
        ベースラインを BlendshapeFrame として返す

        Returns:
            BlendshapeFrame: ニュートラルベースライン
        """
        # emotion_direction.py の BlendshapeFrame を使用
        from emotion_direction import BlendshapeFrame

        if self._data is None:
            return BlendshapeFrame()

        baseline = self._data.get("baseline", {})
        return BlendshapeFrame(values={k: float(v) for k, v in baseline.items()})

    def get_normalize_max(self) -> dict:
        """
        NORMALIZE_MAX 推定値を返す

        Returns:
            dict: {"tension": float, "anger": float,
                   "positive": float, "overall": float}
        """
        if self._data is None:
            # デフォルト値（フォールバック）
            return {
                "tension" : 0.22,
                "anger"   : 0.15,
                "positive": 0.20,
                "overall" : 0.06,
            }
        return self._data.get("normalize_max", {})

    def is_loaded(self) -> bool:
        """データが読み込み済みかを返す"""
        return self._data is not None


# ============================================================
# スタンドアロン実行（CSVからキャリブレーション生成）
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python calibration.py <csv_file1> [csv_file2] ...")
        print("例: python calibration.py frames.csv frames2.csv")
        sys.exit(1)

    builder = CalibrationBuilder()

    for csv_path in sys.argv[1:]:
        builder.load_csv(csv_path)

    cal_data = builder.build()
    builder.save(cal_data)

    print("\n=== キャリブレーション結果 ===")
    print(f"セッション数    : {cal_data.n_sessions}")
    print(f"ベースライン特徴量数: {len(cal_data.baseline)}")
    print(f"NORMALIZE_MAX   : {cal_data.normalize_max}")
    for phase, stats in cal_data.phases.items():
        print(f"  {phase:10s}: {stats.n_frames}フレーム")