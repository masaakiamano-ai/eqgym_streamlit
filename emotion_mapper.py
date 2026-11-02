# ============================================================
# emotion_mapper.py
# FR-06: 52 Blendshape差分値 → 3方向感情スコア変換
#
# 処理フロー:
#   blendshape_delta (dict)
#     → 重み付き線形結合 (weighted sum)
#     → シグモイド正規化 (0〜100)
#     → 移動平均スムージング
#     → emotion_scores (dict)
#
# 出力キー:
#   "tension"           : 生スコア (0〜100)
#   "anger"             : 生スコア (0〜100)
#   "positive"          : 生スコア (0〜100)
#   "smoothed_tension"  : 移動平均スコア (0〜100)
#   "smoothed_anger"    : 移動平均スコア (0〜100)
#   "smoothed_positive" : 移動平均スコア (0〜100)
# ============================================================

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional


# ============================================================
# Blendshape → 感情方向マッピング定義
#
# 各エントリは (blendshape名, 重み) のタプル
# 重み:
#   正の値 → その方向のスコアを上昇させる
#   負の値 → その方向のスコアを抑制する
#
# 重みの大きさは寄与度を表す（1.0=標準、2.0=強い寄与）
#
# 参考: MediaPipe FaceMesh 52 Blendshape名一覧
#   https://developers.google.com/mediapipe/solutions/vision/face_landmarker
# ============================================================

EMOTION_GROUPS: Dict[str, List[Tuple[str, float]]] = {

    # ────────────────────────────────────────────────────────
    # 緊張（Tension）
    # 特徴: 眉間のしわ、目の見開き・緊張、口の結び
    # ────────────────────────────────────────────────────────
    "tension": [
        ("browInnerUp",       +2.0),  # 眉間が上がる（不安・驚き）
        ("browDownLeft",      +1.5),  # 左眉が下がる（緊張・集中）
        ("browDownRight",     +1.5),  # 右眉が下がる
        ("eyeWideLeft",       +1.5),  # 左目が見開く（緊張・驚き）
        ("eyeWideRight",      +1.5),  # 右目が見開く
        ("eyeSquintLeft",     +1.0),  # 左目を細める（集中・緊張）
        ("eyeSquintRight",    +1.0),  # 右目を細める
        ("jawForward",        +1.0),  # 顎が前に出る（緊張）
        ("mouthPressLeft",    +1.5),  # 左口を結ぶ（緊張・我慢）
        ("mouthPressRight",   +1.5),  # 右口を結ぶ
        ("mouthStretchLeft",  +1.0),  # 左口角が横に引く（緊張）
        ("mouthStretchRight", +1.0),  # 右口角が横に引く
        ("mouthClose",        +1.0),  # 口を閉じる
        ("cheekSquintLeft",   -1.0),  # 左頬が上がる → 緊張緩和
        ("cheekSquintRight",  -1.0),  # 右頬が上がる → 緊張緩和
    ],

    # ────────────────────────────────────────────────────────
    # 怒り・圧力（Anger / Pressure）
    # 特徴: 眉が強く下がる、鼻のしわ、口の強張り
    # ────────────────────────────────────────────────────────
    "anger": [
        ("browDownLeft",       +2.0),  # 左眉が強く下がる（怒り）
        ("browDownRight",      +2.0),  # 右眉が強く下がる
        ("noseSneerLeft",      +2.0),  # 左鼻にしわ（嫌悪・怒り）
        ("noseSneerRight",     +2.0),  # 右鼻にしわ
        ("eyeSquintLeft",      +1.5),  # 左目を細める（怒り）
        ("eyeSquintRight",     +1.5),  # 右目を細める
        ("jawOpen",            +1.0),  # 顎が開く（怒鳴る）
        ("mouthFrownLeft",     +1.5),  # 左口角が下がる（不満・怒り）
        ("mouthFrownRight",    +1.5),  # 右口角が下がる
        ("mouthUpperUpLeft",   +1.0),  # 左上唇が上がる（軽蔑）
        ("mouthUpperUpRight",  +1.0),  # 右上唇が上がる
        ("cheekPuff",          +1.0),  # 頬が膨らむ（怒り・息を止める）
        ("mouthSmileLeft",     -1.5),  # 左笑顔 → 怒り緩和
        ("mouthSmileRight",    -1.5),  # 右笑顔 → 怒り緩和
    ],

    # ────────────────────────────────────────────────────────
    # ポジティブ（Positive）
    # 特徴: 笑顔、頬の上昇、目の細まり（デュシェンヌスマイル）
    # ────────────────────────────────────────────────────────
    "positive": [
        ("mouthSmileLeft",    +2.5),  # 左口角が上がる（笑顔）
        ("mouthSmileRight",   +2.5),  # 右口角が上がる
        ("cheekSquintLeft",   +2.0),  # 左頬が上がる（本物の笑顔）
        ("cheekSquintRight",  +2.0),  # 右頬が上がる
        ("eyeSquintLeft",     +1.0),  # 左目が細まる（笑顔）
        ("eyeSquintRight",    +1.0),  # 右目が細まる
        ("mouthDimpleLeft",   +1.5),  # 左えくぼ
        ("mouthDimpleRight",  +1.5),  # 右えくぼ
        ("jawOpen",           +0.5),  # 口が開く（笑い）
        ("browInnerUp",       -0.5),  # 眉間上昇 → ポジティブ低減
        ("browDownLeft",      -1.0),  # 左眉下降 → ポジティブ低減
        ("browDownRight",     -1.0),  # 右眉下降 → ポジティブ低減
        ("mouthFrownLeft",    -1.5),  # 左口角下降 → ポジティブ低減
        ("mouthFrownRight",   -1.5),  # 右口角下降 → ポジティブ低減
    ],
}

# 方向名リスト（順序固定）
DIRECTION_NAMES: List[str] = list(EMOTION_GROUPS.keys())


# ============================================================
# EmotionScoreCalculator
# ============================================================

class EmotionScoreCalculator:
    """
    52 Blendshape差分値から3方向感情スコアを算出するクラス

    Parameters
    ----------
    smoothing_window : int
        移動平均のフレーム数（デフォルト10 ≒ 2秒@5fps）
    sensitivity : float
        シグモイド関数の感度パラメータ
        大きいほど小さな変化にも敏感に反応する（デフォルト3.0）
    emotion_groups : dict, optional
        カスタムグループを指定する場合（デフォルトはEMOTION_GROUPS）

    使用例
    ------
    calc = EmotionScoreCalculator(smoothing_window=10, sensitivity=3.0)
    scores = calc.calculate(blendshape_delta)
    # scores["smoothed_tension"] → 0〜100
    """

    def __init__(
        self,
        smoothing_window: int = 10,
        sensitivity: float = 3.0,
        emotion_groups: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    ):
        self.smoothing_window = smoothing_window
        self.sensitivity = sensitivity
        self.emotion_groups = emotion_groups or EMOTION_GROUPS

        # 方向ごとのスムージングバッファ
        self.buffers: Dict[str, deque] = {
            direction: deque(maxlen=smoothing_window)
            for direction in self.emotion_groups.keys()
        }

        # 統計追跡（デバッグ・チューニング用）
        self._frame_count: int = 0
        self._last_raw_scores: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ──────────────────────────────────────────────────────

    def calculate(
        self,
        blendshape_delta: Dict[str, float],
    ) -> Dict[str, float]:
        """
        1フレーム分のBlendshape差分値から感情スコアを計算する

        Parameters
        ----------
        blendshape_delta : dict
            {blendshape名: 差分値(Current - Baseline)} の辞書
            値の範囲: 通常 -1.0 〜 +1.0

        Returns
        -------
        dict
            {
              "tension"           : float (0〜100),
              "anger"             : float (0〜100),
              "positive"          : float (0〜100),
              "smoothed_tension"  : float (0〜100),
              "smoothed_anger"    : float (0〜100),
              "smoothed_positive" : float (0〜100),
            }
        """
        result: Dict[str, float] = {}

        for direction, feature_weights in self.emotion_groups.items():

            # Step1: 重み付き線形結合
            raw = self._weighted_sum(blendshape_delta, feature_weights)

            # Step2: シグモイド正規化 → 0〜100
            normalized = self._sigmoid_normalize(raw)

            # Step3: バッファに追加
            self.buffers[direction].append(normalized)

            # Step4: 移動平均
            smoothed = float(np.mean(self.buffers[direction]))

            # 結果格納
            result[direction] = round(normalized, 2)
            result[f"smoothed_{direction}"] = round(smoothed, 2)

        # 内部統計更新
        self._frame_count += 1
        self._last_raw_scores = {
            d: result[d] for d in self.emotion_groups.keys()
        }

        return result

    def reset_buffers(self):
        """
        スムージングバッファをリセットする
        新しいセッション開始時に呼び出す
        """
        for direction in self.buffers:
            self.buffers[direction].clear()
        self._frame_count = 0
        self._last_raw_scores = {}

    def get_buffer_fill_rate(self) -> float:
        """
        バッファの充填率を返す（0.0〜1.0）
        起動直後はバッファが埋まっていないため参考値として使用
        """
        if not self.buffers:
            return 0.0
        first_buf = next(iter(self.buffers.values()))
        return len(first_buf) / self.smoothing_window

    def get_dominant_direction(
        self,
        scores: Dict[str, float],
    ) -> str:
        """
        3方向のうち最も高いスコアの方向名を返す

        Parameters
        ----------
        scores : dict
            calculate()の戻り値

        Returns
        -------
        str : "tension" | "anger" | "positive"
        """
        smoothed = {
            d: scores.get(f"smoothed_{d}", 0.0)
            for d in self.emotion_groups.keys()
        }
        return max(smoothed, key=smoothed.get)

    # ──────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ──────────────────────────────────────────────────────

    def _weighted_sum(
        self,
        blendshape_delta: Dict[str, float],
        feature_weights: List[Tuple[str, float]],
    ) -> float:
        """
        重み付き線形結合を計算する

        存在しないBlendshape名は 0.0 として扱う
        重みの絶対値合計で正規化してスケールを統一する
        """
        total = 0.0
        weight_abs_sum = 0.0

        for bs_name, weight in feature_weights:
            value = blendshape_delta.get(bs_name, 0.0)
            total += value * weight
            weight_abs_sum += abs(weight)

        if weight_abs_sum == 0.0:
            return 0.0

        return total / weight_abs_sum

    def _sigmoid_normalize(self, raw: float) -> float:
        """
        シグモイド関数で 0〜100 にマッピングする

        変換式:
            sigmoid(x) = 1 / (1 + exp(-k * x))
            score = sigmoid(raw) * 100

        特性:
            raw = 0.0  → score = 50.0（中立）
            raw > 0.0  → score > 50.0（活性化）
            raw < 0.0  → score < 50.0（抑制）

        Parameters
        ----------
        raw : float
            重み付き線形結合の出力値

        Returns
        -------
        float : 0.0〜100.0
        """
        # オーバーフロー防止のためクリッピング
        clipped = np.clip(raw * self.sensitivity, -500, 500)
        sigmoid = 1.0 / (1.0 + np.exp(-clipped))
        return float(sigmoid * 100.0)


# ============================================================
# ユーティリティ関数
# ============================================================

def get_blendshape_coverage() -> Dict[str, List[str]]:
    """
    各方向でカバーされているBlendshape名の一覧を返す
    デバッグ・確認用
    """
    coverage = {}
    for direction, features in EMOTION_GROUPS.items():
        coverage[direction] = [name for name, _ in features]
    return coverage


def get_all_used_blendshapes() -> List[str]:
    """
    EMOTION_GROUPSで使用されている全Blendshape名（重複なし）を返す
    """
    names = set()
    for features in EMOTION_GROUPS.values():
        for name, _ in features:
            names.add(name)
    return sorted(list(names))


if __name__ == "__main__":
    # ── 動作確認用テスト ──────────────────────────────────
    print("=== EmotionScoreCalculator テスト ===\n")

    calc = EmotionScoreCalculator(smoothing_window=5, sensitivity=3.0)

    # テストケース1: 中立状態（差分ゼロ）
    delta_neutral = {}
    scores = calc.calculate(delta_neutral)
    print("【中立状態】")
    for k, v in scores.items():
        print(f"  {k}: {v:.1f}")

    # テストケース2: 笑顔状態
    delta_smile = {
        "mouthSmileLeft":  0.8,
        "mouthSmileRight": 0.8,
        "cheekSquintLeft": 0.6,
        "cheekSquintRight": 0.6,
    }
    scores = calc.calculate(delta_smile)
    print("\n【笑顔状態】")
    for k, v in scores.items():
        print(f"  {k}: {v:.1f}")

    # テストケース3: 怒り状態
    delta_anger = {
        "browDownLeft":  0.7,
        "browDownRight": 0.7,
        "noseSneerLeft": 0.5,
        "noseSneerRight": 0.5,
    }
    scores = calc.calculate(delta_anger)
    print("\n【怒り状態】")
    for k, v in scores.items():
        print(f"  {k}: {v:.1f}")

    print("\n【使用Blendshape一覧】")
    print(get_all_used_blendshapes())