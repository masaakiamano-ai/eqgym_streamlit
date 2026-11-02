# ============================================================
# ui_eq_display.py
# 概要: EQコンディションスコアの Streamlit UI 表示コンポーネント
#
# 表示構成:
#   Section 1 : EQコンディションスコア（3列カード + プログレスバー）
#   Section 2 : 4方向バランス（Plotly レーダーチャート + 数値補足）
#   Section 3 : スコア計算内訳（エクスパンダー）
#   Section 4 : デバッグ詳細テーブル（show_debug=True 時のみ）
#
# 依存:
#   eq_score.py   : EQScores データクラス
#   streamlit     : UI フレームワーク
#   plotly        : レーダーチャート描画
#   pandas        : テーブル表示
#
# 後方互換性:
#   EQScores に source_phase / reflection_score フィールドが
#   存在しない旧バージョンでも動作する（hasattr でガード）
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from eq_score import EQScores


# ============================================================
# EQスコア表示設定
# ※ key は eq_score.py の EQScores フィールド名と完全一致させること
# ============================================================
EQ_METRICS = [
    {
        "key"  : "awareness_score",       # EQScores.awareness_score
        "label": "気づき",
        "icon" : "🔍",
        "color": "#79c0ff",               # 水色
        "desc" : "自分の感情状態をどれだけ把握できているか",
    },
    {
        "key"  : "regulation_score",      # EQScores.regulation_score
        "label": "感情安定度",
        "icon" : "🎛️",
        "color": "#56d364",               # 緑
        "desc" : "感情の波をコントロールできているか",
    },
    {
        "key"  : "social_margin_score",   # EQScores.social_margin_score
        "label": "対人余白度",
        "icon" : "🤝",
        "color": "#f0883e",               # オレンジ
        "desc" : "他者と関わる余裕・ポジティブ資源の豊かさ",
    },
]

# ── 4方向カラー定義 ──────────────────────────────────────────
DIRECTION_COLORS = {
    "tension"  : "#ff7b72",   # 赤:   緊張
    "anger"    : "#d2a8ff",   # 紫:   怒り
    "positive" : "#f0883e",   # オレンジ: ポジティブ
    "stable"   : "#56d364",   # 緑:   安定
}


# ============================================================
# Section 1: メイン表示関数（3列カード）
# ============================================================

def render_eq_scores(
    eq         : "EQScores | None",
    show_debug : bool = False,
) -> None:
    """
    EQコンディションスコアを3列カードで表示する

    Args:
        eq         : calc_eq_scores() が返した EQScores
                     None の場合は警告を表示して処理を中断する
        show_debug : True のとき Section 4 デバッグテーブルを表示する
    """

    # ── Section ヘッダー ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 EQコンディションスコア ✨")

    # ── NoneType ガード ──────────────────────────────────────
    # eq が None の場合はここで処理を止め、クラッシュを防ぐ
    if eq is None:
        st.warning("⚠️ EQスコアの計算に失敗しました。もう一度計測してください。")
        return

    # ── source_phase の表示（後方互換: フィールドがなければスキップ） ──
    # どのフェーズのデータを使ってEQスコアを算出したか表示する
    if hasattr(eq, "source_phase") and eq.source_phase:
        phase_label = eq.source_phase
        st.caption(f"📍 算出フェーズ: **{phase_label}**")

    # ── 3列カード表示 ────────────────────────────────────────
    cols = st.columns(3)
    for col, metric in zip(cols, EQ_METRICS):

        # getattr の第3引数に 0.0 を指定
        # → 万が一フィールド名が不一致でもクラッシュしない
        score = getattr(eq, metric["key"], 0.0)

        badge_color = _score_to_color(score)
        badge_label = _score_to_label(score)

        with col:
            # ── スコアカード（HTML/CSS）──────────────────────
            st.markdown(f"""
<div style="
    background    : #161b22;
    border        : 1px solid #30363d;
    border-radius : 12px;
    padding       : 1.2rem;
    text-align    : center;
    margin-bottom : 0.5rem;
">
    <div style="font-size:2rem;">{metric['icon']}</div>
    <div style="
        color       : {metric['color']};
        font-size   : 2.2rem;
        font-weight : bold;
        line-height : 1.2;
    ">{score:.0f}</div>
    <div style="color:#8b949e; font-size:0.8rem;">/ 100</div>
    <div style="
        color       : #e6edf3;
        font-size   : 0.95rem;
        font-weight : bold;
        margin-top  : 0.4rem;
    ">{metric['label']}</div>
    <div style="
        background    : {badge_color};
        color         : #0d1117;
        border-radius : 20px;
        padding       : 0.1rem 0.6rem;
        font-size     : 0.75rem;
        font-weight   : bold;
        display       : inline-block;
        margin-top    : 0.3rem;
    ">{badge_label}</div>
</div>""", unsafe_allow_html=True)

            # ── プログレスバー ──────────────────────────────
            # score: 0〜100 → st.progress() 用に 0.0〜1.0 に変換
            st.progress(float(score / 100.0))

            # ── 説明キャプション ────────────────────────────
            st.caption(metric["desc"])

    # ── Section 2: 4方向レーダーチャート ──────────────────────
    st.markdown("### 📡 4方向バランス")
    render_eq_radar_chart(eq)

    # ── Section 3: スコア計算内訳（エクスパンダー） ───────────
    render_eq_radar_text(eq)

    # ── Section 4: デバッグ詳細テーブル（任意表示） ───────────
    if show_debug:
        _render_debug_table(eq)


# ============================================================
# Section 2: 4方向レーダーチャート
# ============================================================

def render_eq_radar_chart(eq: "EQScores | None") -> None:
    """
    4方向比率を Plotly レーダーチャートで表示する

    Args:
        eq : EQScores インスタンス（None の場合は何も表示しない）
    """
    # None ガード
    if eq is None:
        return

    # ── レーダーチャートデータの準備 ──────────────────────────
    # 4方向の比率を % 表示に変換
    categories = [
        "😰 緊張\ntension",
        "😠 怒り\nanger",
        "😊 ポジティブ\npositive",
        "😌 安定\nstable",
    ]
    values = [
        round(eq.p_tension  * 100, 1),
        round(eq.p_anger    * 100, 1),
        round(eq.p_positive * 100, 1),
        round(eq.p_stable   * 100, 1),
    ]

    # レーダーチャートは閉じる必要があるため先頭値を末尾に追加
    values_closed     = values     + [values[0]]
    categories_closed = categories + [categories[0]]

    # ── Plotly レーダーチャート描画 ───────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r             = values_closed,
        theta         = categories_closed,
        fill          = "toself",
        fillcolor     = "rgba(121, 192, 255, 0.15)",   # 水色・半透明
        line          = dict(color="#79c0ff", width=2),
        name          = "4方向比率",
        hovertemplate = "%{theta}: %{r:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        polar = dict(
            bgcolor    = "rgba(22, 27, 34, 0.8)",      # ダーク背景
            radialaxis = dict(
                visible   = True,
                range     = [0, 100],
                tickfont  = dict(size=9, color="#8b949e"),
                gridcolor = "#30363d",
                linecolor = "#30363d",
            ),
            angularaxis = dict(
                tickfont  = dict(size=11, color="#e6edf3"),
                gridcolor = "#30363d",
                linecolor = "#30363d",
            ),
        ),
        showlegend    = False,
        height        = 320,
        margin        = dict(t=30, b=30, l=60, r=60),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── 4方向の数値補足（横並び4列）──────────────────────────
    cols = st.columns(4)
    direction_data = [
        ("😰 緊張",       eq.p_tension  * 100, DIRECTION_COLORS["tension"]),
        ("😠 怒り",       eq.p_anger    * 100, DIRECTION_COLORS["anger"]),
        ("😊 ポジティブ", eq.p_positive * 100, DIRECTION_COLORS["positive"]),
        ("😌 安定",       eq.p_stable   * 100, DIRECTION_COLORS["stable"]),
    ]
    for col, (label, val, color) in zip(cols, direction_data):
        with col:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span style='color:{color}; font-weight:bold; font-size:18px;'>"
                f"{val:.1f}%</span><br>"
                f"<span style='color:#8b949e; font-size:12px;'>{label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# Section 3: スコア計算内訳（エクスパンダー）
# ============================================================

def render_eq_radar_text(eq: "EQScores | None") -> None:
    """
    EQスコアの計算内訳をエクスパンダーで表示する

    Args:
        eq : calc_eq_scores() が返した EQScores
             None の場合は何も表示しない
    """
    # None ガード：eq が None なら何も表示せず終了
    if eq is None:
        return

    with st.expander("📐 スコア計算内訳を見る", expanded=False):

        # ── 4方向比率テーブル ──────────────────────────────────
        st.markdown("#### 入力値（4方向比率）")
        input_df = pd.DataFrame({
            "方向": [
                "😰 緊張",
                "😠 怒り",
                "😊 ポジティブ",
                "😌 安定",
            ],
            "比率": [
                f"{eq.p_tension  * 100:.1f}%",
                f"{eq.p_anger    * 100:.1f}%",
                f"{eq.p_positive * 100:.1f}%",
                f"{eq.p_stable   * 100:.1f}%",
            ],
        })
        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True,
        )

        # ── rawスコアと正規化結果テーブル ──────────────────────
        st.markdown("#### rawスコアと正規化結果")
        st.markdown(f"""
| 指標 | rawスコア | 正規化後（0〜100） |
|:-----|----------:|------------------:|
| 🔍 気づき       | {eq.awareness_raw:.3f}     | **{eq.awareness_score:.1f}**     |
| 🎛️ 感情安定度   | {eq.regulation_raw:.3f}    | **{eq.regulation_score:.1f}**    |
| 🤝 対人余白度   | {eq.social_margin_raw:.3f} | **{eq.social_margin_score:.1f}** |
""")

        # ── 自己申告スコア表示 ─────────────────────────────────
        # reflection_score フィールドが存在する場合のみ表示（後方互換）
        if hasattr(eq, "reflection_score"):
            st.caption(
                f"💡 自己申告スコア（reflection）: "
                f"{eq.reflection_score:.0f} / 100"
            )
        else:
            st.caption("💡 自己申告スコア: 未設定")

        # ── source_phase 表示（後方互換: フィールドがなければスキップ）──
        if hasattr(eq, "source_phase") and eq.source_phase:
            st.caption(f"📍 算出フェーズ: **{eq.source_phase}**")


# ============================================================
# Section 4: デバッグ詳細テーブル（show_debug=True 時のみ）
# ============================================================

def _render_debug_table(eq: "EQScores | None") -> None:
    """
    EQScores の全内部値をデバッグテーブルで表示する

    Args:
        eq : EQScores インスタンス（None の場合は何も表示しない）
    """
    # None ガード
    if eq is None:
        return

    with st.expander("🔍 デバッグ詳細（raw スコア・比率）", expanded=False):

        # ── raw スコアテーブル ────────────────────────────────
        st.markdown("**Raw スコア**")
        raw_df = pd.DataFrame({
            "指標"      : ["awareness", "regulation", "social_margin"],
            "raw 値"    : [
                round(eq.awareness_raw,     4),
                round(eq.regulation_raw,    4),
                round(eq.social_margin_raw, 4),
            ],
            "最終スコア": [
                round(eq.awareness_score,     1),
                round(eq.regulation_score,    1),
                round(eq.social_margin_score, 1),
            ],
        })
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

        # ── 4方向比率テーブル ─────────────────────────────────
        st.markdown("**4方向比率**")
        ratio_df = pd.DataFrame({
            "方向"     : ["tension", "anger", "positive", "stable"],
            "比率 (%)": [
                round(eq.p_tension  * 100, 2),
                round(eq.p_anger    * 100, 2),
                round(eq.p_positive * 100, 2),
                round(eq.p_stable   * 100, 2),
            ],
        })
        st.dataframe(ratio_df, use_container_width=True, hide_index=True)

        # ── その他情報（JSON）────────────────────────────────
        st.markdown("**その他**")
        other_info = {}

        # reflection_score（後方互換）
        if hasattr(eq, "reflection_score"):
            other_info["reflection_score"] = round(eq.reflection_score, 1)

        # source_phase（後方互換）
        if hasattr(eq, "source_phase"):
            other_info["source_phase"] = eq.source_phase

        if other_info:
            st.json(other_info)


# ============================================================
# ヘルパー関数
# ============================================================

def _score_to_label(score: float) -> str:
    """
    スコア値をラベル文字列に変換する

    Args:
        score : 0〜100 のスコア値

    Returns:
        str : "優秀" / "良好" / "普通" / "要注意" / "低め"
    """
    if   score >= 80: return "優秀"
    elif score >= 65: return "良好"
    elif score >= 50: return "普通"
    elif score >= 35: return "要注意"
    else:             return "低め"


def _score_to_color(score: float) -> str:
    """
    スコア値をバッジ背景色（16進数カラーコード）に変換する

    Args:
        score : 0〜100 のスコア値

    Returns:
        str : CSS カラーコード
    """
    if   score >= 80: return "#56d364"   # 緑       ← 優秀
    elif score >= 65: return "#79c0ff"   # 水色     ← 良好
    elif score >= 50: return "#f0883e"   # オレンジ ← 普通
    elif score >= 35: return "#ffa657"   # 薄橙     ← 要注意
    else:             return "#ff7b72"   # 赤       ← 低め