# ============================================================
# continuous_mode_ui.py
# 会議モード（連続計測）のUI画面
# ============================================================
import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

from continuous_recorder import ContinuousRecorder
from eq_score import calc_eq_scores


def render_continuous_mode_page():
    # ── 循環インポート回避のため関数内でインポート ──────────
    from app_main import (
        calc_direction_scores,
        get_blendshapes,
        get_face_landmarker,
        get_camera_thread,
    )

    st.header("🗣️ 会議モード（連続計測）")

    rec = st.session_state.get("continuous_recorder")

    # ── ① 未開始：設定画面 ──────────────────────────
    if rec is None:
        st.subheader("⚙️ 計測設定")

        # TODO: PRESET_CHOICES_MIN が実際に continuous_recorder.py に
        # 定義されている属性名か要確認（未確認のため仮定）
        preset_min = st.selectbox(
            "計測時間（分）",
            getattr(ContinuousRecorder, "PRESET_CHOICES_MIN", [5, 10, 15, 30]),
            index=1,
        )

        baseline = st.session_state.get("baseline")
        baseline_id = st.session_state.get("baseline_id")

        if baseline is None:
            st.error("⚠️ 本日のベースラインが未取得です。先に「🎥 計測」ページでベースラインを取得してください。")
            return

        st.caption(f"使用するbaseline_id: `{baseline_id}`")

        if st.button("▶️ 会議モード開始", type="primary"):
            rec = ContinuousRecorder(
                baseline_id=baseline_id,
                baseline_frame=baseline,   # BlendshapeFrameインスタンス（.get()対応済み想定）
                direction_fn=calc_direction_scores,
                eq_fn=calc_eq_scores,
                preset_min=preset_min,
                window_sec=30,
            )
            st.session_state["continuous_recorder"] = rec
            st.session_state["continuous_summary"] = None
            st.session_state["continuous_saved"] = False
            st.rerun()
        return

    # ── ② 計測中 / ③ 終了後 ─────────────────────────
    summary = st.session_state.get("continuous_summary")
    face_landmarker = get_face_landmarker()
    thread = get_camera_thread()
    frame_bgr = thread.get_frame()          # ★修正: get_frame() が正しいメソッド名

    col_cam, col_info = st.columns([1, 1])

    with col_cam:
        st.subheader("📷 カメラ映像")
        cam_placeholder = st.empty()
        if frame_bgr is not None:
            cam_placeholder.image(frame_bgr, channels="BGR", width=480)
        else:
            cam_placeholder.warning("カメラ映像を取得できません")

    with col_info:
        if summary is None:
            # ── 特徴量抽出 → フレーム記録 ──
            bs_frame = get_blendshapes(frame_bgr, face_landmarker) if frame_bgr is not None else None
            rec.maybe_record_frame(bs_frame)

            st.progress(rec.progress_ratio())
            st.caption(f"経過: {rec.elapsed_sec():.0f}秒 / {rec.preset_sec}秒")

            progress_rows = rec.get_progress_scores()
            if progress_rows:
                st.dataframe(
                    pd.DataFrame(progress_rows),
                    use_container_width=True, hide_index=True,
                )

            if st.button("⏹️ 手動終了"):
                st.session_state["continuous_summary"] = rec.finalize()
                st.rerun()

            if rec.should_auto_stop():
                st.session_state["continuous_summary"] = rec.finalize()
                st.rerun()

        else:
            st.success("計測完了 ✅")
            session_row = summary["session_row"]
            st.write(f"時間: {session_row['duration_sec']}秒 / "
                     f"ウィンドウ数: {session_row['window_count']} / "
                     f"スキップフレーム: {session_row['skipped_frame_count']}")
            st.write(f"EQ: 気づき{session_row['eq_awareness']} / "
                     f"安定{session_row['eq_regulation']} / "
                     f"余白{session_row['eq_social_margin']}")

            notes = st.text_area("📝 メモ（任意）", key="continuous_notes")

            if not st.session_state.get("continuous_saved", False):
                if st.button("💾 保存する", type="primary"):
                    rec.save(notes)
                    st.session_state["continuous_saved"] = True
                    st.success("保存しました")
            else:
                st.info("すでに保存済みです")

            if st.button("🔁 新しい会議モードを開始"):
                st.session_state["continuous_recorder"] = None
                st.session_state["continuous_summary"] = None
                st.session_state["continuous_saved"] = False
                st.rerun()

    if summary is None:
        st_autorefresh(interval=700, key="continuous_refresh")