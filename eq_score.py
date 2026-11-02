# ============================================================
# eq_score.py
# FR-07: 4方向スコア → EQコンディションスコア変換モジュール
# 仕様: EQ GYM MVP Requirements スコア計算式に準拠
#
# 改善④: neutral フェーズのみを使用する calc_eq_scores_from_neutral()
#         を追加。smile/angry/brow_up フェーズの影響を排除し、
#         EQ スコアの安定性を向上させる。
#
# 改善⑤: awareness_score の計算から reflection_score（自己申告スコア）
#         のブレンドを除外。reflection_score は現状未実装のため
#         常に固定値50.0が使われており、実質的にノイズとして
#         awareness_score を歪めていたため。
#         → awareness_score は awareness_expr_score（表情由来）のみで算出。
#         reflection_score は将来のユーザーフィードバック機能実装時に
#         再導入予定。引数・フィールドは互換性のため維持する。
# ============================================================

from dataclasses import dataclass, field
import numpy as np


# ============================================================
# EQScores データクラス
# ============================================================
@dataclass
class EQScores:
    """
    EQコンディションスコアの結果格納クラス

    Attributes:
        awareness_score      : 気づきスコア (0〜100)
                                ※ 表情由来スコアのみで算出（reflection_score未使用）
        regulation_score     : 感情安定度 (0〜100)
        social_margin_score  : 対人余白度 (0〜100)
        p_tension             : 緊張の比率 (0〜1)
        p_anger                : 怒りの比率 (0〜1)
        p_positive             : ポジティブの比率 (0〜1)
        p_stable                : 安定の比率 (0〜1)
        awareness_raw          : 気づき rawスコア
        regulation_raw         : 感情安定度 rawスコア
        social_margin_raw      : 対人余白度 rawスコア
        reflection_score       : 自己申告スコア (0〜100) デフォルト50
                                  ※ 現在未使用（計算には反映されない）。
                                    デバッグ表示・将来機能用にフィールドのみ保持。
        source_phase            : スコア算出に使用したフェーズ名
    """
    # ── 3指標スコア（メイン出力） ──────────────────────────
    awareness_score      : float  # 気づきスコア (0〜100)
    regulation_score     : float  # 感情安定度 (0〜100)
    social_margin_score  : float  # 対人余白度 (0〜100)

    # ── 4方向比率（内部計算値・デバッグ用） ────────────────
    p_tension  : float  # 緊張比率 (0〜1)
    p_anger    : float  # 怒り比率 (0〜1)
    p_positive : float  # ポジティブ比率 (0〜1)
    p_stable   : float  # 安定比率 (0〜1)

    # ── rawスコア（内部計算値・デバッグ用） ────────────────
    awareness_raw      : float  # 気づき raw (1.0〜2.0)
    regulation_raw     : float  # 感情安定度 raw (-3.0〜+3.0)
    social_margin_raw  : float  # 対人余白度 raw (-3.0〜+3.0)

    # ── 自己申告スコア（デフォルト値あり・現在未使用） ─────
    reflection_score : float = 50.0  # NOTE: 計算には未使用。将来機能用に保持。

    # ── スコア算出に使用したフェーズ名（デバッグ・表示用） ──
    source_phase : str = "unknown"  # 例: "neutral", "all"


# ============================================================
# EQスコア計算: 汎用版（全フェーズ対応）
# ============================================================
def calc_eq_scores(
    tension           : float,
    anger             : float,
    positive          : float,
    stable            : float,
    reflection_score  : float = 50.0,
    source_phase      : str = "all",
) -> EQScores:
    """
    4方向スコアからEQコンディションスコアを計算する（汎用版）

    Args:
        tension            : 緊張スコア (0〜1)
        anger              : 怒り・圧力スコア (0〜1)
        positive           : ポジティブスコア (0〜1)
        stable             : 安定スコア (0〜1)
        reflection_score   : 自己申告スコア (0〜100) デフォルト50
                             ※ 現在未使用。互換性のため引数は維持。
        source_phase       : 使用フェーズ名 デフォルト"all"

    Returns:
        EQScores: 3指標のEQスコアと内部計算値をすべて含むデータクラス
        ※ この関数は必ず EQScores を返す（None にならない）
    """
    return _compute_eq(
        tension           = tension,
        anger             = anger,
        positive          = positive,
        stable            = stable,
        reflection_score  = reflection_score,
        source_phase      = source_phase,
    )


# ============================================================
# EQスコア計算: neutral フェーズ特化版（改善④）
# ============================================================
def calc_eq_scores_from_neutral(
    neutral_scores    : dict,
    reflection_score   : float = 50.0,
) -> EQScores:
    """
    neutral フェーズのスコアのみを使ってEQスコアを計算する（改善④）

    smile/angry/brow_up フェーズの影響を完全に排除することで、
    EQスコアが感情フェーズの激しい表情に引きずられる問題を解決する。

    Args:
        neutral_scores    : neutral フェーズの4方向スコア辞書
                             例: {"tension": 0.1, "anger": 0.05,
                                  "positive": 0.03, "stable": 0.8}
                             値の範囲は 0〜1（正規化済み）
        reflection_score  : 自己申告スコア (0〜100) デフォルト50
                            ※ 現在未使用。互換性のため引数は維持。

    Returns:
        EQScores: neutral フェーズベースのEQスコア
        ※ この関数は必ず EQScores を返す（None にならない）
    """
    tension  = float(neutral_scores.get("tension", 0.0))
    anger    = float(neutral_scores.get("anger", 0.0))
    positive = float(neutral_scores.get("positive", 0.0))
    stable   = float(neutral_scores.get("stable", 0.0))

    return _compute_eq(
        tension           = tension,
        anger             = anger,
        positive          = positive,
        stable            = stable,
        reflection_score  = reflection_score,
        source_phase      = "neutral",
    )


# ============================================================
# 内部計算関数（共通処理）
# ============================================================
def _compute_eq(
    tension           : float,
    anger             : float,
    positive          : float,
    stable            : float,
    reflection_score  : float,
    source_phase      : str,
) -> EQScores:
    """
    EQスコアの実計算を行う内部関数
    （calc_eq_scores / calc_eq_scores_from_neutral 共用）

    改善⑤: awareness_score は awareness_expr_score（表情由来）のみで
            算出する。reflection_score とのブレンドは行わない。

    Args:
        tension, anger, positive, stable : 各方向スコア (0〜1)
        reflection_score                  : 自己申告スコア (0〜100)
                                             ※ 未使用。EQScores格納用にのみ保持。
        source_phase                       : 使用フェーズ名

    Returns:
        EQScores
    """
    # ── Step 1: 4方向比率化 ──────────────────────────────────
    EPSILON = 1e-8
    total = tension + anger + positive + stable + EPSILON
    p_tension  = tension  / total
    p_anger    = anger    / total
    p_positive = positive / total
    p_stable   = stable   / total

    # ── Step 2: rawスコア計算 ────────────────────────────────
    awareness_raw = (
        1.0 * p_tension
        + 1.0 * p_anger
        + 2.0 * p_positive
        + 2.0 * p_stable
    )

    regulation_raw = (
        -2.0 * p_tension
        + -3.0 * p_anger
        + 2.0 * p_positive
        + 3.0 * p_stable
    )

    social_margin_raw = (
        -1.0 * p_tension
        + -3.0 * p_anger
        + 3.0 * p_positive
        + 2.0 * p_stable
    )

    # ── Step 3: 0〜100 正規化 ────────────────────────────────
    # 【改善⑤】reflection_score とのブレンドを除外。
    #   awareness_score = awareness_expr_score（表情由来のみ）
    #   NOTE: reflection_score は現在未使用（保留中）。
    #         将来のユーザーフィードバック機能実装時に
    #         再導入を検討する。
    awareness_score = 50.0 + 50.0 * (awareness_raw - 1.0)

    regulation_score     = ((regulation_raw + 3.0) / 6.0) * 100.0
    social_margin_score  = ((social_margin_raw + 3.0) / 6.0) * 100.0

    # ── Step 4: 0〜100 クリップ ──────────────────────────────
    awareness_score      = float(np.clip(awareness_score, 0.0, 100.0))
    regulation_score     = float(np.clip(regulation_score, 0.0, 100.0))
    social_margin_score  = float(np.clip(social_margin_score, 0.0, 100.0))

    # ── Step 5: EQScores を構築して返す ─────────────────────
    return EQScores(
        awareness_score      = awareness_score,
        regulation_score     = regulation_score,
        social_margin_score  = social_margin_score,
        p_tension            = p_tension,
        p_anger               = p_anger,
        p_positive             = p_positive,
        p_stable                = p_stable,
        awareness_raw          = awareness_raw,
        regulation_raw          = regulation_raw,
        social_margin_raw       = social_margin_raw,
        reflection_score         = reflection_score,  # 保持のみ（計算には不使用）
        source_phase              = source_phase,
    )