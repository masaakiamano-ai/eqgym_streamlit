# ============================================================
# ui_emotion_display.py
# FR-06: 感情方向スコアのStreamlit表示コンポーネント
#
# 提供コンポーネント:
#   render_score_gauges()      : 3方向ゲージUI
#   render_realtime_chart()    : リアルタイム時系列グラフ
#   render_dominant_badge()    : 支配的方向バッジ
#   render_session_summary()   : セッションサマリー表示
# ============================================================

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, List, Optional


# ============================================================
# 表示設定定数
# ============================================================

DIRECTION_CONFIG: Dict[str, Dict] = {
    "tension": {
        "label":     "緊張",
        "label_jp":  "😰 緊張",
        "color":     "#FF6B6B",
        "bg_color":  "rgba(255,107,107,0.15)",
        "border":    "rgba(255,107,107,0.6)",
        "icon":      "😰",
        "high_msg":  "強い緊張が検出されています",
        "mid_msg":   "やや緊張しています",
        "low_msg":   "リラックスしています",
    },
    "anger": {
        "label":     "怒り・圧力",
        "label_jp":  "😤 怒り・圧力",
        "color":     "#FF4500",
        "bg_color":  "rgba(255,69,0,0.15)",
        "border":    "rgba(255,69,0,0.6)",
        "icon":      "😤",
        "high_msg":  "強い圧力・怒りが検出されています",
        "mid_msg":   "やや圧力がかかっています",
        "low_msg":   "穏やかな状態です",
    },
    "positive": {
        "label":     "ポジティブ",
        "label_jp":  "😊 ポジティブ",
        "color":     "#00D4AA",
        "bg_color":  "rgba(0,212,170,0.15)",
        "border":    "rgba(0,212,170,0.6)",
        "icon":      "😊",
        "high_msg":  "とてもポジティブな状態です",
        "mid_msg":   "ポジティブな傾向があります",
        "low_msg":   "ニュートラルな状態です",
    },
}

# スコアレベル定義
SCORE_LEVELS = [
    (75, "HIGH",  "#FF4444", "🔴"),
    (55, "MID",   "#FFB800", "🟡"),
    (45, "LOW",   "#00D4AA", "🟢"),
    (0,  "CALM",  "#4488FF", "🔵"),
]


# ============================================================
# PUBLIC COMPONENTS
# ============================================================

def render_score_gauges(emotion_scores: Dict[str, float]):
    """
    3方向スコアをカードUIとプログレスバーで表示する

    Parameters
    ----------
    emotion_scores : dict
        EmotionScoreCalculator.calculate()の戻り値
        "smoothed_tension", "smoothed_anger", "smoothed_positive" を使用
    """
    st.markdown(
        "<h3 style='color:white; margin-bottom:12px;'>"
        "🎯 感情方向スコア（リアルタイム）"
        "</h3>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)

    for i, (direction, config) in enumerate(DIRECTION_CONFIG.items()):
        score = emotion_scores.get(f"smoothed_{direction}", 50.0)
        level_label, level_color, level_icon = _get_level(score)
        message = _get_message(score, config)

        with cols[i]:
            # ── カードUI ──
            st.markdown(
                f"""
                <div style="
                    background: {config['bg_color']};
                    border: 1px solid {config['border']};
                    border-radius: 12px;
                    padding: 16px 12px;
                    text-align: center;
                    min-height: 160px;
                ">
                    <div style="font-size: 2.2rem; margin-bottom: 4px;">
                        {config['icon']}
                    </div>
                    <div style="
                        color: {config['color']};
                        font-size: 0.8rem;
                        font-weight: bold;
                        letter-spacing: 0.05em;
                        margin-bottom: 8px;
                    ">
                        {config['label']}
                    </div>
                    <div style="
                        color: white;
                        font-size: 2.8rem;
                        font-weight: bold;
                        line-height: 1;
                        margin-bottom: 4px;
                    ">
                        {score:.0f}
                    </div>
                    <div style="
                        color: {level_color};
                        font-size: 0.75rem;
                        font-weight: bold;
                        margin-bottom: 8px;
                    ">
                        {level_icon} {level_label}
                    </div>
                    <div style="
                        color: rgba(255,255,255,0.6);
                        font-size: 0.7rem;
                        line-height: 1.3;
                    ">
                        {message}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── プログレスバー ──
            progress_val = max(0, min(100, int(score)))
            st.progress(progress_val)


def render_realtime_chart(
    history: Dict[str, deque],
    fps: float = 5.0,
    height: int = 260,
):
    """
    3方向スコアのリアルタイム時系列グラフを表示する

    Parameters
    ----------
    history : dict
        {"smoothed_tension": deque, "smoothed_anger": deque,
         "smoothed_positive": deque}
    fps : float
        フレームレート（X軸を秒換算するために使用）
    height : int
        グラフの高さ（px）
    """
    st.markdown(
        "<h3 style='color:white; margin-bottom:8px;'>"
        "📈 感情変化タイムライン"
        "</h3>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    for direction, config in DIRECTION_CONFIG.items():
        key = f"smoothed_{direction}"
        y_data = list(history.get(key, []))

        if not y_data:
            continue

        # X軸: フレーム → 秒換算
        x_data = [i / fps for i in range(len(y_data))]

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            name=config["label_jp"],
            line=dict(color=config["color"], width=2.0),
            fill="tozeroy",
            fillcolor=config["bg_color"],
            mode="lines",
            hovertemplate=(
                f"<b>{config['label']}</b><br>"
                "時刻: %{x:.1f}秒<br>"
                "スコア: %{y:.1f}<extra></extra>"
            ),
        ))

    # 中立ライン（50）
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="中立",
        annotation_font_color="rgba(255,255,255,0.5)",
        annotation_font_size=10,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
        yaxis=dict(
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.08)",
            title="スコア",
            title_font_size=11,
            tickfont_size=10,
        ),
        xaxis=dict(
            title="経過時間（秒）",
            gridcolor="rgba(255,255,255,0.08)",
            title_font_size=11,
            tickfont_size=10,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font_size=11,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_dominant_badge(emotion_scores: Dict[str, float]):
    """
    現在の支配的感情方向をバッジ表示する

    Parameters
    ----------
    emotion_scores : dict
        EmotionScoreCalculator.calculate()の戻り値
    """
    # 最高スコアの方向を特定
    smoothed = {
        d: emotion_scores.get(f"smoothed_{d}", 50.0)
        for d in DIRECTION_CONFIG.keys()
    }
    dominant = max(smoothed, key=smoothed.get)
    dominant_score = smoothed[dominant]
    config = DIRECTION_CONFIG[dominant]

    st.markdown(
        f"""
        <div style="
            background: {config['bg_color']};
            border: 2px solid {config['color']};
            border-radius: 8px;
            padding: 10px 16px;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        ">
            <span style="font-size: 1.5rem;">{config['icon']}</span>
            <div>
                <div style="
                    color: rgba(255,255,255,0.7);
                    font-size: 0.7rem;
                ">現在の支配的感情</div>
                <div style="
                    color: {config['color']};
                    font-size: 1.1rem;
                    font-weight: bold;
                ">
                    {config['label']} ({dominant_score:.0f})
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_session_summary(summary: Dict):
    """
    セッションサマリー統計をテーブルで表示する

    Parameters
    ----------
    summary : dict
        SessionDataManager.get_session_stats()の戻り値
    """
    if not summary or summary.get("status") == "no_data":
        st.info("セッションデータがありません")
        return

    st.markdown(
        "<h3 style='color:white;'>📊 セッションサマリー</h3>",
        unsafe_allow_html=True,
    )

    # 基本情報
    col1, col2 = st.columns(2)
    with col1:
        st.metric("総フレーム数", f"{summary.get('total_frames', 0)} frames")
    with col2:
        st.metric(
            "セッション時間",
            f"{summary.get('duration_sec', 0):.1f} 秒",
        )

    # 支配的方向
    dominant = summary.get("dominant_direction", "N/A")
    if dominant in DIRECTION_CONFIG:
        config = DIRECTION_CONFIG[dominant]
        st.markdown(
            f"**支配的感情方向:** "
            f"{config['icon']} {config['label']}",
        )

    # スコア統計テーブル
    smoothed_scores = summary.get("smoothed_scores", {})
    if smoothed_scores:
        rows = []
        for direction, stats in smoothed_scores.items():
            config = DIRECTION_CONFIG.get(direction, {})
            rows.append({
                "方向":   config.get("label_jp", direction),
                "平均":   f"{stats.get('mean', 0):.1f}",
                "最大":   f"{stats.get('max', 0):.1f}",
                "最小":   f"{stats.get('min', 0):.1f}",
                "標準偏差": f"{stats.get('std', 0):.1f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_buffer_status(fill_rate: float):
    """
    スムージングバッファの充填状態を表示する
    起動直後のウォームアップ期間に表示

    Parameters
    ----------
    fill_rate : float
        0.0〜1.0 (EmotionScoreCalculator.get_buffer_fill_rate()の戻り値)
    """
    if fill_rate >= 1.0:
        return  # 満タンなら非表示

    pct = int(fill_rate * 100)
    st.markdown(
        f"""
        <div style="
            background: rgba(255,184,0,0.15);
            border: 1px solid rgba(255,184,0,0.5);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 0.8rem;
            color: #FFB800;
        ">
            ⏳ ウォームアップ中... ({pct}%) 
            スムージングバッファを充填しています
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PRIVATE HELPERS
# ============================================================

def _get_level(score: float):
    """
    スコアに応じたレベルラベル・色・アイコンを返す

    Returns
    -------
    tuple[str, str, str]
        (label, color, icon)
    """
    for threshold, label, color, icon in SCORE_LEVELS:
        if score >= threshold:
            return label, color, icon
    return "CALM", "#4488FF", "🔵"


def _get_message(score: float, config: Dict) -> str:
    """
    スコアに応じたメッセージを返す
    """
    if score >= 75:
        return config.get("high_msg", "")
    elif score >= 55:
        return config.get("mid_msg", "")
    else:
        return config.get("low_msg", "")


# ============================================================
# 動作確認（Streamlit単体実行用）
# ============================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="UI Emotion Display Test",
        layout="wide",
    )
    st.title("🧪 ui_emotion_display.py テスト")

    from collections import deque
    import random

    # ダミースコア生成
    dummy_scores = {
        "tension":           45.0,
        "anger":             30.0,
        "positive":          72.0,
        "smoothed_tension":  48.0,
        "smoothed_anger":    32.0,
        "smoothed_positive": 70.0,
    }

    # ダミー履歴生成
    dummy_history = {
        "smoothed_tension":  deque(
            [40 + random.gauss(0, 5) for _ in range(50)], maxlen=150
        ),
        "smoothed_anger":    deque(
            [30 + random.gauss(0, 4) for _ in range(50)], maxlen=150
        ),
        "smoothed_positive": deque(
            [65 + random.gauss(0, 6) for _ in range(50)], maxlen=150
        ),
    }

    render_buffer_status(0.6)
    render_dominant_badge(dummy_scores)
    st.divider()
    render_score_gauges(dummy_scores)
    render_realtime_chart(dummy_history)