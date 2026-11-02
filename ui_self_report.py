# ============================================================
# ui_self_report.py
# 役割：計測終了後に自己申告スコアを入力するUIを提供する
#       入力値は data/labels/self_reports.csv に保存される
# ============================================================

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

LABELS_DIR       = Path("data") / "labels"
SELF_REPORTS_CSV = LABELS_DIR / "self_reports.csv"

LABELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# render_self_report()
# 計測結果画面の下部に呼び出す
# 引数:
#   session_id : str  紐付けるセッションID
# ============================================================
def render_self_report(session_id: str):

    st.markdown("---")
    st.subheader("📝 セルフチェック（自己申告）")
    st.caption(
        "計測結果と照らし合わせて、今の状態を教えてください。"
        "このデータが個人最適化の学習データになります。"
    )

    # ── スライダー入力 ─────────────────────────────────────────
    with st.form(key=f"self_report_{session_id}"):

        col1, col2 = st.columns(2)

        with col1:
            tension = st.slider(
                "😰 緊張度",
                min_value=0, max_value=10, value=5,
                help="計測中どのくらい緊張していましたか？（0=全くない、10=非常に強い）"
            )
            emotion_var = st.slider(
                "🌊 感情の揺れ",
                min_value=0, max_value=10, value=5,
                help="感情が揺れやすい状態でしたか？（0=安定、10=揺れやすい）"
            )

        with col2:
            social_margin = st.slider(
                "🤝 対人余裕",
                min_value=0, max_value=10, value=5,
                help="人と接する余裕がありましたか？（0=全くない、10=十分ある）"
            )
            overall = st.slider(
                "☀️ 全体コンディション",
                min_value=0, max_value=10, value=5,
                help="今日の全体的な状態は？（0=最悪、10=最高）"
            )

        notes = st.text_area(
            "📌 メモ（任意）",
            placeholder="気になったこと、体調、環境など自由に記入してください",
            height=80
        )

        submitted = st.form_submit_button(
            "💾 自己申告を保存",
            use_container_width=True
        )

    # ── 保存処理 ───────────────────────────────────────────────
    if submitted:
        _save_self_report(
            session_id   = session_id,
            tension      = tension,
            emotion_var  = emotion_var,
            social_margin= social_margin,
            overall      = overall,
            notes        = notes
        )
        st.success("✅ 自己申告を保存しました！データが学習に活用されます。")

        # 入力値のサマリーを表示
        st.markdown(
            f"""
            | 項目 | 入力値 |
            |------|--------|
            | 😰 緊張度 | {tension} / 10 |
            | 🌊 感情の揺れ | {emotion_var} / 10 |
            | 🤝 対人余裕 | {social_margin} / 10 |
            | ☀️ 全体コンディション | {overall} / 10 |
            """
        )


# ============================================================
# _save_self_report()  ← 内部関数
# 自己申告データを CSV に追記保存する
# ============================================================
def _save_self_report(
    session_id: str,
    tension: int,
    emotion_var: int,
    social_margin: int,
    overall: int,
    notes: str
):
    new_row = pd.DataFrame([{
        "session_id"    : session_id,
        "reported_at"   : datetime.now().isoformat(),
        "self_tension"  : tension,
        "self_emotion_var"   : emotion_var,
        "self_social_margin" : social_margin,
        "self_overall"       : overall,
        "notes"              : notes,
    }])

    if SELF_REPORTS_CSV.exists():
        df_existing = pd.read_csv(SELF_REPORTS_CSV)
        new_row = pd.concat([df_existing, new_row], ignore_index=True)

    new_row.to_csv(SELF_REPORTS_CSV, index=False)
    print(f"[SelfReport] 保存完了: session_id={session_id}")