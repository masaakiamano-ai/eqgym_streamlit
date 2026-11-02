# ============================================================
# ui_history.py
# FR-08: 履歴閲覧 UI
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_manager import load_summary, load_session_json, delete_session


def render_history_page():
    """履歴ページ全体をレンダリングする"""

    st.markdown("## 📈 計測履歴")

    df = load_summary()

    # ── データなし ────────────────────────────────────────────
    if df.empty:
        st.info("📭 まだ計測データがありません。計測を完了すると履歴が表示されます。")
        return

    # ── サマリーカード ────────────────────────────────────────
    _render_summary_cards(df)

    st.divider()

    # ── トレンドグラフ ────────────────────────────────────────
    _render_trend_chart(df)

    st.divider()

    # ── 方向比率グラフ ────────────────────────────────────────
    _render_ratio_chart(df)

    st.divider()

    # ── 履歴テーブル ──────────────────────────────────────────
    _render_history_table(df)


# ── サマリーカード ────────────────────────────────────────────
def _render_summary_cards(df: pd.DataFrame):
    """直近・平均・最高スコアをカード表示"""

    latest = df.iloc[-1]
    cols = st.columns(4)

    eq_cols = [
        ("awareness_score",     "🧠 気づき"),
        ("regulation_score",    "🌊 感情安定"),
        ("social_margin_score", "🤝 対人余白"),
        ("reflection_score",    "🔍 自己省察"),
    ]

    for col, (key, label) in zip(cols, eq_cols):
        val     = latest[key]
        avg_val = df[key].mean()
        delta   = val - avg_val
        col.metric(
            label     = label,
            value     = f"{val:.1f}",
            delta     = f"{delta:+.1f} vs 平均",
            delta_color = "normal",
        )


# ── EQスコア トレンドグラフ ────────────────────────────────────
def _render_trend_chart(df: pd.DataFrame):
    """EQスコアの時系列折れ線グラフ"""

    st.markdown("### 📉 EQスコア トレンド")

    eq_metrics = {
        "awareness_score":     ("🧠 気づき",     "#4FC3F7"),
        "regulation_score":    ("🌊 感情安定",   "#81C784"),
        "social_margin_score": ("🤝 対人余白",   "#FFB74D"),
        "reflection_score":    ("🔍 自己省察",   "#CE93D8"),
    }

    fig = go.Figure()

    for key, (label, color) in eq_metrics.items():
        fig.add_trace(go.Scatter(
            x          = df["timestamp"],
            y          = df[key],
            mode       = "lines+markers",
            name       = label,
            line       = dict(color=color, width=2),
            marker     = dict(size=6),
            hovertemplate = f"{label}: %{{y:.1f}}<br>%{{x}}<extra></extra>",
        ))

    fig.update_layout(
        yaxis      = dict(range=[0, 100], title="スコア"),
        xaxis      = dict(title="計測日時"),
        legend     = dict(orientation="h", y=-0.2),
        height     = 350,
        margin     = dict(l=40, r=20, t=20, b=60),
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font       = dict(color="#EEEEEE"),
    )

    st.plotly_chart(fig, use_container_width=True)


# ── 方向比率 トレンドグラフ ────────────────────────────────────
def _render_ratio_chart(df: pd.DataFrame):
    """4方向比率の時系列グラフ"""

    st.markdown("### 🧭 感情方向比率 トレンド")

    ratio_metrics = {
        "tension_ratio":  ("⚡ 緊張",     "#EF5350"),
        "anger_ratio":    ("😠 怒り",     "#FF7043"),
        "positive_ratio": ("😊 ポジティブ", "#66BB6A"),
        "stable_ratio":   ("😌 安定",     "#42A5F5"),
    }

    fig = go.Figure()

    for key, (label, color) in ratio_metrics.items():
        fig.add_trace(go.Scatter(
            x          = df["timestamp"],
            y          = df[key],
            mode       = "lines+markers",
            name       = label,
            line       = dict(color=color, width=2),
            marker     = dict(size=6),
            hovertemplate = f"{label}: %{{y:.3f}}<br>%{{x}}<extra></extra>",
        ))

    fig.update_layout(
        yaxis      = dict(title="比率"),
        xaxis      = dict(title="計測日時"),
        legend     = dict(orientation="h", y=-0.2),
        height     = 300,
        margin     = dict(l=40, r=20, t=20, b=60),
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font       = dict(color="#EEEEEE"),
    )

    st.plotly_chart(fig, use_container_width=True)


# ── 履歴テーブル ──────────────────────────────────────────────
def _render_history_table(df: pd.DataFrame):
    """履歴一覧テーブルと削除機能"""

    st.markdown("### 🗂️ 計測履歴一覧")

    # 表示用に整形
    display_df = df[[
        "timestamp", "session_id",
        "awareness_score", "regulation_score",
        "social_margin_score", "reflection_score",
        "memo"
    ]].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    display_df.columns = [
        "計測日時", "セッションID",
        "🧠気づき", "🌊安定", "🤝余白", "🔍省察",
        "メモ"
    ]

    st.dataframe(
        display_df.sort_values("計測日時", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # ── 削除機能 ──────────────────────────────────────────────
    with st.expander("🗑️ セッション削除"):
        session_ids = df["session_id"].tolist()
        target = st.selectbox("削除するセッションID", session_ids)
        if st.button("削除する", type="secondary"):
            delete_session(target)
            st.success(f"✅ セッション {target} を削除しました")
            st.rerun()

    # ── CSV エクスポート ───────────────────────────────────────
    st.download_button(
        label     = "📥 CSV ダウンロード",
        data      = df.to_csv(index=False).encode("utf-8-sig"),
        file_name = "eq_history.csv",
        mime      = "text/csv",
    )