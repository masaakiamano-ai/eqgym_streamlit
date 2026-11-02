# ============================================================
# emotion_direction.py
# 概要: MediaPipe Blendshapeから4方向の感情スコアを算出する
#
# 4方向:
#   1. tension  (緊張方向)
#   2. anger    (怒り・圧力方向)
#   3. positive (ポジティブ方向)
#   4. stable   (安定方向)
#
# 処理フロー:
#   Blendshape値 → Delta計算 → 4方向スコア → 0〜1正規化
#   → 方向比率 → EQ指標スコア（FR-07用）
#
# 改善②: smile_suppression を anger 計算に追加
#   笑顔時に browDown が連動上昇することで anger が
#   誤検出される問題を smile 強度で抑制する
#
# 改善③: stable スコアを neutral フェーズ特化で再設計
#   overall_change の上限を下げ、stable が高く出やすくする
#
# 注意:
#   本スコアはEQ能力診断ではなく、
#   Neutral Baselineからの表情変化に基づく参考指標である。
# ============================================================

import numpy as np                          # 数値計算・配列操作
from dataclasses import dataclass, field    # 構造化データクラス
from typing import Optional                 # 型ヒント（Optional型）


# ============================================================
# Blendshapeキー定義
# 概要: MediaPipe Face Landmarker が返す52個のBlendshapeキー名
# ============================================================
BLENDSHAPE_KEYS = [
    # ── 眉 ─────────────────────────────────────────────────
    "browDownLeft",       # 左眉を下げる
    "browDownRight",      # 右眉を下げる
    "browInnerUp",        # 眉の内側を上げる（驚き・心配）
    "browOuterUpLeft",    # 左眉の外側を上げる
    "browOuterUpRight",   # 右眉の外側を上げる
    # ── 目 ─────────────────────────────────────────────────
    "eyeBlinkLeft",       # 左目のまばたき
    "eyeBlinkRight",      # 右目のまばたき
    "eyeLookDownLeft",    # 左目を下に向ける
    "eyeLookDownRight",   # 右目を下に向ける
    "eyeLookInLeft",      # 左目を内側に向ける
    "eyeLookInRight",     # 右目を内側に向ける
    "eyeLookOutLeft",     # 左目を外側に向ける
    "eyeLookOutRight",    # 右目を外側に向ける
    "eyeLookUpLeft",      # 左目を上に向ける
    "eyeLookUpRight",     # 右目を上に向ける
    "eyeSquintLeft",      # 左目を細める（笑顔・集中）
    "eyeSquintRight",     # 右目を細める
    "eyeWideLeft",        # 左目を見開く（驚き・緊張）
    "eyeWideRight",       # 右目を見開く
    # ── 顎・顔 ─────────────────────────────────────────────
    "jawForward",         # 顎を前に出す（怒り・圧力）
    "jawLeft",            # 顎を左にずらす
    "jawOpen",            # 口を開ける（驚き・緊張）
    "jawRight",           # 顎を右にずらす
    # ── 口・唇 ─────────────────────────────────────────────
    "mouthClose",         # 口を閉じる
    "mouthDimpleLeft",    # 左口角のえくぼ
    "mouthDimpleRight",   # 右口角のえくぼ
    "mouthFrownLeft",     # 左口角を下げる（悲しみ）
    "mouthFrownRight",    # 右口角を下げる
    "mouthFunnel",        # 口をすぼめてファンネル形
    "mouthLeft",          # 口全体を左にずらす
    "mouthLowerDownLeft", # 下唇左を下げる
    "mouthLowerDownRight",# 下唇右を下げる
    "mouthPressLeft",     # 左唇を押しつける（怒り・緊張）
    "mouthPressRight",    # 右唇を押しつける
    "mouthPucker",        # 口をすぼめる
    "mouthRight",         # 口全体を右にずらす
    "mouthRollLower",     # 下唇を内側に巻く
    "mouthRollUpper",     # 上唇を内側に巻く
    "mouthShrugLower",    # 下唇を前に突き出す
    "mouthShrugUpper",    # 上唇を前に突き出す
    "mouthSmileLeft",     # 左口角を上げる（笑顔）
    "mouthSmileRight",    # 右口角を上げる
    "mouthStretchLeft",   # 左口角を横に引く
    "mouthStretchRight",  # 右口角を横に引く
    "mouthUpperUpLeft",   # 上唇左を上げる
    "mouthUpperUpRight",  # 上唇右を上げる
    # ── 鼻 ─────────────────────────────────────────────────
    "noseSneerLeft",      # 左鼻孔を広げる（軽蔑・怒り）
    "noseSneerRight",     # 右鼻孔を広げる
    # ── 頬 ─────────────────────────────────────────────────
    "cheekPuff",          # 頬を膨らませる
    "cheekSquintLeft",    # 左頬を上げる（笑顔補助）
    "cheekSquintRight",   # 右頬を上げる
    # ── 舌 ─────────────────────────────────────────────────
    "tongueOut",          # 舌を出す
]


# ============================================================
# 4方向スコアを構成するBlendshapeグループ定義
# ============================================================

# 緊張方向: browInnerUp_delta, eyeWide_delta, jawOpen_delta
TENSION_KEYS = [
    "browInnerUp",   # 眉内側を上げる → 心配・驚き・緊張
    "eyeWideLeft",   # 左目を見開く   → 驚き・緊張
    "eyeWideRight",  # 右目を見開く   → 驚き・緊張
    "jawOpen",       # 口を開ける     → 驚き・緊張
]

# 怒り・圧力方向: browDown_delta, jawForward_delta, mouthPress_delta
ANGER_KEYS = [
    "browDownLeft",    # 左眉を下げる → 怒り・不満
    "browDownRight",   # 右眉を下げる → 怒り・不満
    "jawForward",      # 顎を前に出す → 圧力・攻撃性
    "mouthPressLeft",  # 左唇を押しつける → 緊張・怒り
    "mouthPressRight", # 右唇を押しつける → 緊張・怒り
    "noseSneerLeft",   # 左鼻孔拡張 → 軽蔑・怒り ★追加
    "noseSneerRight",  # 右鼻孔拡張 → 軽蔑・怒り ★追加
    "mouthFrownLeft",  # 左口角下げ → 不満・怒り ★追加
    "mouthFrownRight", # 右口角下げ → 不満・怒り ★追加
]

# ポジティブ方向: mouthSmile_delta, cheekRaise_delta, eyeSquint_delta
POSITIVE_KEYS = [
    "mouthSmileLeft",  # 左口角を上げる → 笑顔
    "mouthSmileRight", # 右口角を上げる → 笑顔
    "cheekSquintLeft", # 左頬を上げる   → デュシェンヌスマイル
    "cheekSquintRight",# 右頬を上げる   → デュシェンヌスマイル
    "eyeSquintLeft",   # 左目を細める   → 笑顔時の目の変化
    "eyeSquintRight",  # 右目を細める   → 笑顔時の目の変化
]


# ============================================================
# データクラス定義
# ============================================================

@dataclass
class BlendshapeFrame:
    """
    1フレーム分のBlendshape値を保持するデータクラス

    Attributes:
        values: キー名 → 値(0.0〜1.0) の辞書
    """
    values: dict = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        """
        指定キーの値を取得する（存在しない場合はデフォルト値）

        Args:
            key    : Blendshapeキー名
            default: キーが存在しない場合の戻り値

        Returns:
            float: Blendshape値（0.0〜1.0）
        """
        return float(self.values.get(key, default))


@dataclass
class DirectionScores:
    """
    4方向スコアを保持するデータクラス
    全スコアは 0.0〜1.0 の範囲に正規化される

    Attributes:
        tension : 緊張方向スコア
        anger   : 怒り・圧力方向スコア
        positive: ポジティブ方向スコア
        stable  : 安定方向スコア
        ratio   : 4方向の比率辞書（合計1.0）
        raw     : 正規化前の生スコア辞書
    """
    tension : float = 0.0
    anger   : float = 0.0
    positive: float = 0.0
    stable  : float = 0.0
    ratio   : dict  = field(default_factory=dict)
    raw     : dict  = field(default_factory=dict)

    def dominant(self) -> str:
        """
        最も強い方向を返す

        Returns:
            str: "tension" / "anger" / "positive" / "stable"
        """
        scores = {
            "tension" : self.tension,
            "anger"   : self.anger,
            "positive": self.positive,
            "stable"  : self.stable,
        }
        return max(scores, key=scores.get)

    def to_dict(self) -> dict:
        """
        スコアを辞書形式で返す（CSV保存・表示用）

        Returns:
            dict: 全スコアと比率を含む辞書
        """
        return {
            "tension"        : round(self.tension,  4),
            "anger"          : round(self.anger,    4),
            "positive"       : round(self.positive, 4),
            "stable"         : round(self.stable,   4),
            "dominant"       : self.dominant(),
            "ratio_tension"  : round(self.ratio.get("tension",  0.0), 4),
            "ratio_anger"    : round(self.ratio.get("anger",    0.0), 4),
            "ratio_positive" : round(self.ratio.get("positive", 0.0), 4),
            "ratio_stable"   : round(self.ratio.get("stable",   0.0), 4),
        }


# ============================================================
# EmotionDirectionCalculator クラス
# ============================================================

class EmotionDirectionCalculator:
    """
    Blendshapeのデルタ値から4方向感情スコアを算出するクラス

    処理フロー:
        1. Delta計算: current - baseline（負値はクリップ）
        2. tension  スコア計算（smile補正あり）
        3. anger    スコア計算（smile_suppression 追加 ← 改善②）
        4. positive スコア計算
        5. stable   計算: 1 - normalize(overall_change)  ← 改善③
        6. 0〜1正規化
        7. 方向比率計算（合計1.0）
        8. スムージング（指数移動平均）
    """

    # ── 正規化上限値 ──────────────────────────────────────────
    NORMALIZE_MAX = {
        "tension" : 0.22,   # 【v4】0.20 → 0.22
        "anger"   : 0.15,   # 【v3】0.20 → 0.15
        "positive": 0.20,   # 【v3】0.25 → 0.20
        "overall" : 0.06,   # 【v4】0.08 → 0.06（stable を上げやすく）
    }

    def set_normalize_max(self, normalize_max: dict) -> None:
        """
        calibration.py から取得した NORMALIZE_MAX を設定する

        Args:
            normalize_max: {"tension": float, "anger": float,
                            "positive": float, "overall": float}
        """
        for key in ["tension", "anger", "positive", "overall"]:
            if key in normalize_max:
                self.NORMALIZE_MAX[key] = float(normalize_max[key])

        print(f"[EmotionDirectionCalculator] "
              f"NORMALIZE_MAX更新: {self.NORMALIZE_MAX}")
    
    # ── スムージング係数 ──────────────────────────────────────
    SMOOTH_ALPHA = 0.4

    # ── smile_suppression パラメータ（改善②） ────────────────
    # anger の誤検出を smile 強度で抑制する
    #   suppression_factor = max(SUPPRESS_MIN, 1.0 - smile * SUPPRESS_RATE)
    #   → smile=0.0 のとき factor=1.0（抑制なし）
    #   → smile=0.5 のとき factor=0.65（35%抑制）
    #   → smile=1.0 のとき factor=0.3 （70%抑制）
    SUPPRESS_RATE : float = 0.7   # 抑制強度（大きいほど強く抑制）
    SUPPRESS_MIN  : float = 0.3   # 抑制係数の最小値（完全ゼロにはしない）

    # ── stable スコア上限（改善③） ────────────────────────────
    # neutral フェーズで stable が適切に高く出るよう上限を緩和
    STABLE_CAP : float = 0.70     # 【v4】0.50 → 0.70

    def __init__(self):
        """初期化: スムージング用の前フレームスコアをリセット"""
        self._prev_ratios: Optional[dict] = None


    # ==========================================================
    # メインメソッド: compute()
    # ==========================================================

    def compute(
        self,
        current : BlendshapeFrame,
        baseline: BlendshapeFrame,
    ) -> dict:
        """
        4方向スコアを計算する

        改善②: anger 計算に smile_suppression を追加
            笑顔時に browDown が連動上昇して anger が
            誤検出される問題を smile 強度で抑制する

        改善③: stable スコアの上限を緩和（0.50 → 0.70）
            neutral フェーズで stable が適切に高く出るよう調整

        Args:
            current : 現在フレームの BlendshapeFrame
            baseline: ニュートラル基準の BlendshapeFrame

        Returns:
            dict: {
                "scores" : {tension, anger, positive, stable} (0〜100),
                "ratios" : {tension, anger, positive, stable} (合計100),
                "dominant": 最も強い方向名
            }
        """

        # ── Step 1: デルタ計算 ────────────────────────────────
        # current - baseline（負値はゼロクリップ）
        delta = self._calc_all_deltas(current, baseline)

        # ── Step 2: smile 強度を取得（改善②で使用） ──────────
        # 左右の mouthSmile デルタ値の平均
        smile_strength = (
            delta.get("mouthSmileLeft",  0.0) +
            delta.get("mouthSmileRight", 0.0)
        ) / 2.0

        # ── Step 3a: tension スコア計算（smile補正あり） ──────
        tension_vals    = [delta.get(k, 0.0) for k in TENSION_KEYS]
        # TENSION_KEYS順: [browInnerUp, eyeWideLeft, eyeWideRight, jawOpen]
        tension_weights = [1.0, 1.0, 1.0, 1.0]

        if smile_strength >= 0.3:
            # smile が強いほど browInnerUp・eyeWide の重みを下げる
            # jawOpen（index=3）は補正しない（本物の驚き・緊張に対応）
            smile_factor       = max(0.3, 1.0 - smile_strength)  # 0.3〜0.7
            tension_weights[0] *= smile_factor  # browInnerUp
            tension_weights[1] *= smile_factor  # eyeWideLeft
            tension_weights[2] *= smile_factor  # eyeWideRight

        w_sum       = sum(tension_weights)
        tension_raw = (
            sum(v * w for v, w in zip(tension_vals, tension_weights)) / w_sum
        )

        # ── Step 3b: anger スコア計算（smile_suppression 追加）──
        #
        # 【改善②のコア処理】
        # 問題: 笑顔時に口角上昇と連動して browDown も微上昇する
        #       → anger が誤って高くなる
        #
        # 解決: smile_strength に応じて anger_raw を減衰させる
        #
        # 計算式:
        #   suppression_factor = max(SUPPRESS_MIN, 1.0 - smile * SUPPRESS_RATE)
        #   anger_raw_suppressed = anger_raw * suppression_factor
        #
        anger_raw = self._mean_delta(delta, ANGER_KEYS)

        # smile_suppression 係数を計算（0.3〜1.0）
        suppression_factor = max(
            self.SUPPRESS_MIN,                          # 最低 0.3 は残す
            1.0 - smile_strength * self.SUPPRESS_RATE, # smile が強いほど小さく
        )

        # anger_raw に抑制係数を乗算
        anger_raw_suppressed = anger_raw * suppression_factor

        # ── Step 3c: positive スコア計算（補正なし） ──────────
        positive_raw = self._mean_delta(delta, POSITIVE_KEYS)

        # ── Step 4: 各スコアを 0〜1 に正規化 ─────────────────
        tension_norm  = self._normalize(tension_raw,          self.NORMALIZE_MAX["tension"])
        anger_norm    = self._normalize(anger_raw_suppressed, self.NORMALIZE_MAX["anger"])
        positive_norm = self._normalize(positive_raw,         self.NORMALIZE_MAX["positive"])

        # ── Step 5: stable スコア計算（改善③） ───────────────
        #
        # 【改善③のコア処理】
        # overall_change: 3方向の生スコアの平均（全体的な表情変化量）
        # anger は suppression 後の値を使用（smile 時の過大評価を防ぐ）
        #
        # overall_max を 0.06 に下げることで:
        #   → neutral 時の微小変化でも stable が高く出やすくなる
        #   → 表情変化が大きいフェーズでは stable が低くなる
        #
        overall_change = float(np.mean([
            tension_raw,
            anger_raw_suppressed,  # ← suppression 済みの値を使用
            positive_raw,
        ]))

        # stable = 1 - normalize(overall_change)
        # STABLE_CAP（0.70）でキャップ → neutral 時に適切に高く出る
        stable_norm = min(
            1.0 - self._normalize(overall_change, self.NORMALIZE_MAX["overall"]),
            self.STABLE_CAP,  # 【v4】0.50 → 0.70
        )

        # ── Step 6: 比率計算（合計1.0） ───────────────────────
        raw_scores = {
            "tension" : tension_norm,
            "anger"   : anger_norm,
            "positive": positive_norm,
            "stable"  : stable_norm,
        }
        ratios = self._calc_ratio(raw_scores)

        # ── Step 7: スムージング（指数移動平均） ──────────────
        smoothed = self._smooth(ratios)

        # ── Step 8: 結果を辞書形式で返す ──────────────────────
        return {
            "scores": {
                k: round(v * 100, 1)
                for k, v in raw_scores.items()
            },
            "ratios": {
                k: round(v * 100, 1)
                for k, v in smoothed.items()
            },
            "dominant": max(smoothed, key=smoothed.get),
        }


    # ==========================================================
    # プライベートメソッド群
    # ==========================================================

    def _calc_all_deltas(
        self,
        current : BlendshapeFrame,
        baseline: BlendshapeFrame,
    ) -> dict:
        """
        全52 Blendshapeのデルタ値を計算する
        delta = max(0, current - baseline)

        Args:
            current : 現在フレームの BlendshapeFrame
            baseline: ベースラインの BlendshapeFrame

        Returns:
            dict: キー → delta値（0.0以上）の辞書
        """
        deltas = {}
        for key in BLENDSHAPE_KEYS:
            cur_val  = current.get(key,  0.0)
            base_val = baseline.get(key, 0.0)
            deltas[key] = max(0.0, cur_val - base_val)  # 負値はゼロクリップ
        return deltas

    def _mean_delta(self, all_deltas: dict, keys: list) -> float:
        """
        指定キーグループのデルタ値の平均を計算する

        Args:
            all_deltas: 全Blendshapeのdelta辞書
            keys      : 平均を取るBlendshapeキーのリスト

        Returns:
            float: 指定グループのdelta平均値
        """
        values = [all_deltas.get(k, 0.0) for k in keys]
        if not values:
            return 0.0
        return float(np.mean(values))

    def _normalize(self, value: float, max_val: float) -> float:
        """
        値を 0〜1 に正規化する（max_val でクリップ）

        Args:
            value  : 正規化対象の値
            max_val: 正規化の上限値（これ以上は1.0になる）

        Returns:
            float: 0.0〜1.0 の正規化済み値
        """
        if max_val <= 0.0:
            return 0.0
        return float(np.clip(value / max_val, 0.0, 1.0))

    def _calc_ratio(self, scores: dict) -> dict:
        """
        4方向スコアから方向比率を計算する（合計1.0）
        stable の比率上限を 0.65 に制限（改善③に合わせて緩和）

        Args:
            scores: {"tension": float, "anger": float,
                     "positive": float, "stable": float}

        Returns:
            dict: 各方向の比率（合計1.0）
        """
        capped = dict(scores)
        # stable の比率上限: 【v4】0.55 → 0.65（STABLE_CAP に合わせて緩和）
        if capped.get("stable", 0.0) > 0.65:
            capped["stable"] = 0.65

        total = sum(capped.values())
        if total <= 0.0:
            n = len(capped)
            return {k: 1.0 / n for k in capped}
        return {k: v / total for k, v in capped.items()}

    def _smooth(self, current_ratios: dict) -> dict:
        """
        前フレームとの指数移動平均でスムージングする
        smoothed = alpha * prev + (1 - alpha) * current

        Args:
            current_ratios: 現在フレームの比率辞書

        Returns:
            dict: スムージング済み比率辞書（合計1.0）
        """
        if self._prev_ratios is None:
            self._prev_ratios = current_ratios.copy()
            return current_ratios.copy()

        smoothed = {}
        for k in current_ratios:
            prev_val    = self._prev_ratios.get(k, current_ratios[k])
            current_val = current_ratios[k]
            smoothed[k] = (
                self.SMOOTH_ALPHA       * prev_val +
                (1.0 - self.SMOOTH_ALPHA) * current_val
            )

        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        self._prev_ratios = smoothed.copy()
        return smoothed

    def reset(self):
        """スムージング状態をリセットする"""
        self._prev_ratios = None


# ============================================================
# ユーティリティ関数
# ============================================================

def blendshape_list_to_frame(bs_list: list) -> BlendshapeFrame:
    """
    MediaPipe の face_blendshapes 結果リストを BlendshapeFrame に変換する

    Args:
        bs_list: MediaPipe face_blendshapes のカテゴリリスト

    Returns:
        BlendshapeFrame
    """
    values = {}
    for item in bs_list:
        values[item.category_name] = float(item.score)
    return BlendshapeFrame(values=values)


def blendshape_dict_to_frame(bs_dict: dict) -> BlendshapeFrame:
    """
    辞書形式のBlendshapeデータを BlendshapeFrame に変換する

    Args:
        bs_dict: キー名 → 値の辞書

    Returns:
        BlendshapeFrame
    """
    return BlendshapeFrame(
        values={k: float(v) for k, v in bs_dict.items()}
    )


def average_frames(frames: list) -> BlendshapeFrame:
    """
    複数フレームのBlendshapeFrameを平均して1フレームにまとめる

    Args:
        frames: BlendshapeFrame のリスト

    Returns:
        BlendshapeFrame: 全フレームの平均値を持つフレーム
    """
    if not frames:
        return BlendshapeFrame()

    all_keys = set()
    for f in frames:
        all_keys.update(f.values.keys())

    avg_values = {}
    for key in all_keys:
        vals = [f.get(key, 0.0) for f in frames]
        avg_values[key] = float(np.mean(vals))

    return BlendshapeFrame(values=avg_values)


def result_to_direction_scores(result: dict) -> DirectionScores:
    """
    EmotionDirectionCalculator.compute() の戻り値 dict を
    DirectionScores データクラスに変換する

    Args:
        result: compute() が返す dict

    Returns:
        DirectionScores
    """
    scores = result.get("scores", {})
    ratios = result.get("ratios", {})

    return DirectionScores(
        tension  = scores.get("tension",  0.0) / 100.0,
        anger    = scores.get("anger",    0.0) / 100.0,
        positive = scores.get("positive", 0.0) / 100.0,
        stable   = scores.get("stable",   0.0) / 100.0,
        ratio = {
            "tension"  : ratios.get("tension",  0.0) / 100.0,
            "anger"    : ratios.get("anger",    0.0) / 100.0,
            "positive" : ratios.get("positive", 0.0) / 100.0,
            "stable"   : ratios.get("stable",   0.0) / 100.0,
        },
        raw = scores,
    )