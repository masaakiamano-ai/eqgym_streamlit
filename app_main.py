# ============================================================
# app_main.py
# 概要: EQ GYM メインアプリケーション
#
# FR-06: 4方向スコア（tension / anger / positive / stable）
# FR-07: EQコンディションスコア（awareness / regulation / social_margin）
# FR-08: データ保存・履歴管理
#   - data_recorder.py : 生データ parquet 保存
#   - ui_self_report.py : 自己申告 UI
#   - data_manager.py  : SessionData JSON/CSV 保存
#   - ui_history.py    : 履歴ページ
#
# 改善①: EQ計算を neutral フェーズのみのデータで実施
# 改善②: MediaPipe Tasks FaceLandmarker による 52 blendshape 取得
# 改善③: baseline_id による日次ベースライン管理
#   - 1日1回（複数回可・手動再取得対応）ベースラインを取得
#   - baseline_scores.csv に baseline_id, date, recorded_at, is_manual を記録
#   - sessions.csv に baseline_id を記録し紐付ける
# 改善④: ベースライン重複保存防止ガード（baseline_saved_once）
# 改善⑤: get_camera_thread() の冗長呼び出しを削除
# 改善⑥: ML方向スコアモデルの本組み込み
# 改善⑦: 会議モード（連続計測）をサイドバーに独立ページとして追加  ★NEW
# ============================================================

# ── 標準ライブラリ・外部ライブラリ ──────────────────────────
import streamlit as st
import cv2
import numpy as np
import csv
import datetime
import time
import os
import threading
import queue

# ── MediaPipe Tasks API ──────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# ── emotion_direction（ルールベース計算：フォールバック用に維持）──
from emotion_direction import (
    EmotionDirectionCalculator,
    BlendshapeFrame,
    DirectionScores,
    blendshape_list_to_frame,
    average_frames,
    result_to_direction_scores,
)

# ── MLモデル（方向スコア推論：本組み込み） ───────────────────
from ml_direction_model import predict_direction_scores as ml_predict_direction_scores

# ── FR-07: EQスコア ──────────────────────────────────────────
from eq_score import (
    calc_eq_scores,
    EQScores,
)
from ui_eq_display import render_eq_scores

# ── FR-08: データ保存・履歴 ──────────────────────────────────
from data_manager import build_session, save_session
import ui_history

# ── FR-08 追加: DataRecorder & SelfReport ────────────────────
from data_recorder import DataRecorder
from ui_self_report import render_self_report

# ── baseline_id 対応: 日次ベースライン管理 ───────────────────
from baseline_store import get_today_latest_baseline, save_baseline_entry

# ── 会議モード（連続計測）UI ★NEW ────────────────────────────
# ※ continuous_mode_ui.py 側は関数内で `from app_main import ...` を
#   遅延importしているため、ここでのトップレベルimportは循環importに
#   ならない（app_mainの読み込みが完了してから呼び出されるため）。
import continuous_mode_ui


# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="EQ GYM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# カスタム CSS
# ============================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color : #0d1117;
    color            : #e6edf3;
    font-family      : 'Hiragino Sans', 'Yu Gothic', sans-serif;
}
.main-header {
    background    : linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border        : 1px solid #30363d;
    border-radius : 12px;
    padding       : 1.5rem 2rem;
    margin-bottom : 1.5rem;
}
.main-title    { color: #ff6eb4; font-size: 1.8rem; font-weight: bold; }
.main-subtitle { color: #8b949e; font-size: 0.95rem; }
.step-card {
    background    : #161b22;
    border        : 1px solid #30363d;
    border-radius : 10px;
    padding       : 1.2rem 1.5rem;
    margin-bottom : 1rem;
}
.info-box {
    background    : #0d2137;
    border        : 1px solid #1f6feb;
    border-radius : 8px;
    padding       : 1rem 1.2rem;
    margin-bottom : 0.8rem;
    color         : #79c0ff;
}
.success-box {
    background    : #0d2818;
    border        : 1px solid #238636;
    border-radius : 8px;
    padding       : 1rem 1.2rem;
    margin-bottom : 0.8rem;
    color         : #56d364;
}
.metric-card {
    background    : #161b22;
    border        : 1px solid #30363d;
    border-radius : 10px;
    padding       : 1rem;
    text-align    : center;
}
.save-box {
    background    : #1a1a2e;
    border        : 1px solid #7c3aed;
    border-radius : 10px;
    padding       : 1.2rem 1.5rem;
    margin-top    : 1rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 定数定義
# ============================================================
SCORES_CSV = "scores_v3.csv"              # 計測スコア保存ファイル
BASELINE_SEC = 5                          # ベースライン計測秒数
PHASE_SEC = 5                             # 各フェーズ計測秒数
FACE_LANDMARKER_MODEL = "face_landmarker.task"  # モデルファイルパス

# ── スコアリング方式切り替えフラグ（ロールバック用） ─────────
SCORING_MODE = "ml"

# ── 分類クラス → 方向軸 マッピング（確定仕様・参照用） ───────
PHASE_TO_AXIS = {
    "smile": "positive",
    "brow_up": "tension",
    "angry": "anger",
    "neutral": "stable",
}

# ============================================================
# 計測フェーズ定義（4フェーズ）
# ============================================================
PHASES = [
    {
        "key": "smile",
        "label": "😊 スマイル",
        "instruction": "自然な笑顔をしてください 😊",
        "target_axis": "positive",
        "tip": "口角を上げて、目も一緒に細めると効果的です",
    },
    {
        "key": "brow_up",
        "label": "😮 眉上げ",
        "instruction": "眉をできるだけ高く上げてください 😮",
        "target_axis": "tension",
        "tip": "目を大きく見開くように意識すると検出されやすいです",
    },
    {
        "key": "angry",
        "label": "😠 怒り顔",
        "instruction": "眉をしかめて怒った表情をしてください 😠",
        "target_axis": "anger",
        "tip": "眉間にしわを寄せ、口を軽く閉じて力を入れてください",
    },
    {
        "key": "neutral",
        "label": "😐 無表情",
        "instruction": "リラックスして無表情を保ってください 😐",
        "target_axis": "stable",
        "tip": "ベースライン計測と同じ状態をキープしてください",
    },
]

# ============================================================
# 4方向スコア表示設定
# ============================================================
DIRECTION_CONFIG = {
    "tension":  {"label": "緊張",     "icon": "😰", "color": "#f0883e"},
    "anger":    {"label": "怒り・圧力", "icon": "😠", "color": "#ff7b72"},
    "positive": {"label": "ポジティブ", "icon": "😊", "color": "#56d364"},
    "stable":   {"label": "安定",     "icon": "😌", "color": "#79c0ff"},
}
AXES = ["tension", "anger", "positive", "stable"]

# ============================================================
# EmotionDirectionCalculator グローバルインスタンス（フォールバック用）
# ============================================================
_direction_calc = EmotionDirectionCalculator()


# ============================================================
# CameraThread クラス
# ============================================================
class CameraThread:
    """バックグラウンドでカメラフレームを取得し続けるスレッド"""

    def __init__(self, camera_index: int = 0):
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread = None
        self.camera_index = camera_index

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # 鏡像
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
            time.sleep(0.033)  # 約30fps

        cap.release()


# ============================================================
# カメラスレッド管理
# ============================================================
def get_camera_thread() -> CameraThread:
    if "cam_thread" not in st.session_state:
        st.session_state.cam_thread = CameraThread(0)
    ct: CameraThread = st.session_state.cam_thread
    if ct._thread is None or not ct._thread.is_alive():
        ct.start()
    return ct


def stop_camera_thread():
    if "cam_thread" in st.session_state:
        st.session_state.cam_thread.stop()
        del st.session_state["cam_thread"]


# ============================================================
# MediaPipe Tasks FaceLandmarker（キャッシュ）
# ============================================================
@st.cache_resource
def get_face_landmarker() -> FaceLandmarker:
    """
    FaceLandmarker を IMAGE モードで初期化してキャッシュ返却。
    face_landmarker.task がプロジェクトルートに存在すること。
    """
    base_options = mp_python.BaseOptions(
        model_asset_path=FACE_LANDMARKER_MODEL
    )
    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


# ============================================================
# Blendshape フレーム取得（52 blendshape 直接取得版）
# ============================================================
def get_blendshapes(
    frame: np.ndarray,
    face_landmarker: FaceLandmarker,
) -> "BlendshapeFrame | None":
    """
    BGR フレームを受け取り、FaceLandmarker で検出した
    52 blendshape を BlendshapeFrame として返す。
    顔未検出の場合は None を返す。
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )
    detection_result = face_landmarker.detect(mp_image)

    if (
        not detection_result.face_blendshapes
        or len(detection_result.face_blendshapes) == 0
    ):
        return None

    raw_blendshapes = detection_result.face_blendshapes[0]
    values: dict[str, float] = {
        bs.category_name: float(bs.score)
        for bs in raw_blendshapes
        if bs.category_name != "_neutral"
    }
    return BlendshapeFrame(values=values)


# ============================================================
# 4方向スコア計算（改善⑥: ML本組み込み + フォールバック）
# ============================================================
def calc_direction_scores(
    current_frame: BlendshapeFrame,
    baseline_frame: "BlendshapeFrame | None",
) -> DirectionScores:
    """
    SCORING_MODE = "ml" の場合、ml_direction_model.py の
    RandomForestモデルで方向スコアを推論する。

    以下の場合は自動的にルールベース（emotion_direction.py）へ
    フォールバックする:
      - SCORING_MODE が "rule_based" の場合
      - baseline_frame が None（ベースライン未取得）の場合
      - ML推論中に例外が発生した場合（モデル欠損・特徴量不整合等）

    quick測定・background測定・会議モードのいずれの呼び出し経路でも
    同一シグネチャで利用できる。
    """
    if SCORING_MODE == "ml" and baseline_frame is not None:
        try:
            return ml_predict_direction_scores(current_frame, baseline_frame)
        except Exception as e:
            print(f"[WARN] ML方向スコア推論に失敗。ルールベースへフォールバック: {e}")

    bf = baseline_frame if baseline_frame is not None else BlendshapeFrame()
    result = _direction_calc.compute(current_frame, bf)
    return result_to_direction_scores(result)


# ============================================================
# カメラ映像オーバーレイ描画
# ============================================================
def draw_overlay(frame: np.ndarray, scores: DirectionScores) -> np.ndarray:
    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (280, 115), (0, 0, 0), -1)
    frame = cv2.addWeighted(ov, 0.45, frame, 0.55, 0)

    for txt, pos in [
        (f"Tension : {scores.tension * 100:.0f}%", (15, 35)),
        (f"Anger   : {scores.anger * 100:.0f}%", (15, 58)),
        (f"Positive: {scores.positive * 100:.0f}%", (15, 81)),
        (f"Stable  : {scores.stable * 100:.0f}%", (15, 104)),
    ]:
        cv2.putText(frame, txt, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
    return frame


# ============================================================
# CSV 保存・読み込み（計測スコアのみ／ベースラインは baseline_store.py に移管）
# ============================================================
def save_scores(results: list):
    fields = ["phase", "tension", "anger", "positive", "stable", "dominant"]
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({
                k: round(r[k], 4) if isinstance(r.get(k), float) else r.get(k, "")
                for k in fields
            })


# ============================================================
# ベースライン状態の初期化（日付が変わったら必須で再取得）
# ============================================================
def init_baseline_state():
    """
    起動時 & 日付変更時にベースライン状態を初期化する。

    仕様（確定版）:
      - 1日1回、ベースライン取得は必須（日付変更時は必ず再取得が必要）
      - 同日内は複数回の再取得を許可（手動再取得）
      - 同日内に複数存在する場合は recorded_at が最新のものを使用する
      - 自動フォールバック（前日のベースライン継続利用）はしない
    """
    today = datetime.date.today().isoformat()
    if st.session_state.get("baseline_date") != today:
        baseline_id, features, recorded_at = get_today_latest_baseline()
        if features is not None:
            st.session_state["baseline_id"] = baseline_id
            st.session_state["baseline"] = BlendshapeFrame(values=features)
            st.session_state["baseline_recorded_at"] = recorded_at
        else:
            st.session_state["baseline_id"] = None
            st.session_state["baseline"] = None
            st.session_state["baseline_recorded_at"] = None
        st.session_state["baseline_date"] = today


# ============================================================
# リセット処理
# ============================================================
def full_reset():
    """セッション状態を全リセットして TOP 画面に戻る"""
    stop_camera_thread()
    keys = [
        "app_stage",
        "baseline_frames",
        "baseline_start",
        "baseline_saved_once",
        "phase_frames",
        "phase_start",
        "phase_results",
        "current_phase",
        "eq_scores",
        "eq_source_label",
        "session_saved",
        "recorder",
        "saved_session_id",
        "force_manual_baseline",
    ]
    for k in keys:
        st.session_state.pop(k, None)
    st.rerun()


# ============================================================
# 結果表示
# ============================================================
def render_results(phase_results: list):
    """
    計測結果画面を描画する

    表示構成:
      ① 4方向スコア平均カード
      ② 方向比率プログレスバー
      ③ FR-07: EQコンディションスコア（neutral フェーズ優先）
      ④ フェーズ別詳細タブ
      ⑤ FR-08: DataRecorder 保存 + data_manager 保存 + SelfReport UI
      ⑥ もう一度計測するボタン
    """
    st.markdown("## 📊 計測結果")

    avg = {
        ax: float(np.mean([r.get(ax, 0.0) for r in phase_results]))
        for ax in AXES
    }

    # ================================================================
    # ① 4方向スコア平均カード
    # ================================================================
    cols = st.columns(4)
    for col, ax in zip(cols, AXES):
        cfg = DIRECTION_CONFIG[ax]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;">{cfg['icon']}</div>
                <div style="color:{cfg['color']};font-size:1.5rem;font-weight:bold;">
                    {avg[ax] * 100:.0f}%
                </div>
                <div style="color:#8b949e;">{cfg['label']}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ================================================================
    # ② 方向比率プログレスバー
    # ================================================================
    total_avg = sum(avg.values())
    ratio = (
        {k: v / total_avg for k, v in avg.items()}
        if total_avg > 0
        else {k: 0.25 for k in avg}
    )

    st.markdown("### 🧭 方向比率（全フェーズ平均）")
    for ax in AXES:
        cfg = DIRECTION_CONFIG[ax]
        c1, c2, c3 = st.columns([2, 6, 2])
        with c1:
            st.markdown(f"{cfg['icon']} {cfg['label']}")
        with c2:
            st.progress(float(ratio[ax]))
        with c3:
            st.markdown(
                f"<span style='color:{cfg['color']};font-weight:bold;'>"
                f"{ratio[ax] * 100:.1f}%</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ================================================================
    # ③ FR-07: EQコンディションスコア
    # ================================================================
    if st.session_state.get("eq_scores") is None:
        neutral_result = next(
            (r for r in phase_results if r.get("phase") == "neutral"),
            None,
        )
        if neutral_result is not None:
            eq_input = neutral_result
            eq_source_label = "😐 neutral フェーズのみ"
        else:
            eq_input = avg
            eq_source_label = "📊 全フェーズ平均（fallback）"

        try:
            eq_obj = calc_eq_scores(
                tension=eq_input.get("tension", 0.0),
                anger=eq_input.get("anger", 0.0),
                positive=eq_input.get("positive", 0.0),
                stable=eq_input.get("stable", 0.0),
                reflection_score=50.0,
            )
            if eq_obj is not None:
                try:
                    eq_obj.source_phase = eq_source_label
                except AttributeError:
                    object.__setattr__(eq_obj, "source_phase", eq_source_label)
                st.session_state["eq_scores"] = eq_obj
                st.session_state["eq_source_label"] = eq_source_label
        except Exception as e:
            st.error(f"❌ EQスコア計算エラー: {e}")
            st.session_state["eq_scores"] = None
            st.session_state["eq_source_label"] = "エラー"

    eq: "EQScores | None" = st.session_state.get("eq_scores")
    source_label = st.session_state.get("eq_source_label", "")
    if source_label and source_label != "エラー":
        st.caption(f"🎯 EQスコア計算基準: **{source_label}**")
    render_eq_scores(eq, show_debug=st.session_state.get("debug_mode", False))

    st.divider()

    # ================================================================
    # ④ フェーズ別詳細タブ
    # ================================================================
    st.markdown("### 🔍 フェーズ別詳細")
    phase_map = {p["key"]: p for p in PHASES}
    tabs = st.tabs([
        phase_map.get(r["phase"], {}).get("label", r["phase"])
        for r in phase_results
    ])

    for tab, r in zip(tabs, phase_results):
        with tab:
            phase_def = phase_map.get(r["phase"], {})
            target_axis = phase_def.get("target_axis", "")
            tip = phase_def.get("tip", "")
            dominant = r.get("dominant", "none")

            if target_axis and dominant == target_axis:
                cfg_t = DIRECTION_CONFIG.get(target_axis, {})
                st.markdown(
                    '<div class="success-box" style="padding:0.5rem;">'
                    f'✅ 期待通りの方向（{cfg_t.get("icon","")} '
                    f'{cfg_t.get("label","")}）が最も高くなりました！'
                    '</div>', unsafe_allow_html=True,
                )
            elif target_axis:
                cfg_t = DIRECTION_CONFIG.get(target_axis, {})
                cfg_d = DIRECTION_CONFIG.get(dominant, {})
                st.markdown(
                    '<div class="info-box" style="padding:0.5rem;">'
                    f'💡 期待: {cfg_t.get("icon","")} {cfg_t.get("label","")}'
                    f' → 実際: {cfg_d.get("icon","")} '
                    f'{cfg_d.get("label", dominant)}'
                    '</div>', unsafe_allow_html=True,
                )

            if tip:
                st.caption(f"💡 ヒント: {tip}")

            for ax in AXES:
                cfg = DIRECTION_CONFIG[ax]
                val = r.get(ax, 0.0)
                marker = " ◀ 主方向" if ax == dominant else ""
                target_m = " ⭐" if ax == target_axis else ""
                c1, c2, c3 = st.columns([2, 5, 2])
                with c1:
                    st.markdown(f"{cfg['icon']} {cfg['label']}{target_m}")
                with c2:
                    st.progress(float(val))
                with c3:
                    st.markdown(
                        f"<span style='color:{cfg['color']};font-weight:bold;'>"
                        f"{int(val * 100)}%{marker}</span>",
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ================================================================
    # ⑤ FR-08: データ保存セクション
    # ================================================================
    st.markdown("### 💾 この計測を保存する")

    if st.session_state.get("saved_session_id") is not None:
        sid = st.session_state["saved_session_id"]
        st.markdown(
            '<div class="success-box">'
            f'✅ 保存済み（session_id: <code>{sid[:8]}...</code>）<br>'
            '履歴ページで確認できます。'
            '</div>',
            unsafe_allow_html=True,
        )
        render_self_report(sid)
    else:
        st.markdown('<div class="save-box">', unsafe_allow_html=True)
        memo = st.text_input(
            "📝 メモ（任意）",
            placeholder="例: 朝の計測、少し緊張していた",
            key="save_memo",
        )
        if st.button("💾 保存する", type="primary", key="btn_save"):
            current_eq = st.session_state.get("eq_scores")
            if current_eq is None:
                st.warning(
                    "⚠️ EQスコアが計算されていません。"
                    "しばらく待ってから再試行してください。"
                )
            else:
                neutral_result = next(
                    (r for r in phase_results if r.get("phase") == "neutral"),
                    None,
                )
                ratio_source = neutral_result if neutral_result is not None else avg
                ratios = {
                    "tension":  ratio_source.get("tension", 0.0),
                    "anger":    ratio_source.get("anger", 0.0),
                    "positive": ratio_source.get("positive", 0.0),
                    "stable":   ratio_source.get("stable", 0.0),
                }
                raw_phase_data = {
                    r["phase"]: {k: v for k, v in r.items() if k != "phase"}
                    for r in phase_results
                }

                try:
                    recorder: "DataRecorder | None" = st.session_state.get("recorder")
                    if recorder is not None:
                        sid = recorder.save_session(
                            direction_scores=ratios,
                            eq_scores={
                                "awareness":     current_eq.awareness_score,
                                "regulation":    current_eq.regulation_score,
                                "social_margin": current_eq.social_margin_score,
                            },
                            baseline_id=st.session_state.get("baseline_id"),
                            notes=memo,
                        )
                    else:
                        import uuid
                        sid = str(uuid.uuid4())
                        st.warning(
                            "⚠️ DataRecorder が見つかりません。"
                            "フレームデータなしで保存します。"
                        )

                    session = build_session(
                        ratios=ratios,
                        eq_scores=current_eq,
                        raw_phase_data=raw_phase_data,
                        memo=memo,
                    )
                    saved_path = save_session(session)

                    st.session_state["saved_session_id"] = sid
                    st.session_state["session_saved"] = True
                    st.success(f"✅ 保存完了: `{saved_path}`")
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 保存エラー: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ================================================================
    # ⑥ もう一度計測するボタン
    # ================================================================
    if st.button("🔄 もう一度計測する", type="primary"):
        full_reset()


# ============================================================
# メイン関数
# ============================================================
def main():
    # ── ヘッダー ─────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🧠 EQ GYM</div>
        <div class="main-subtitle">感情知性トレーニングシステム</div>
    </div>""", unsafe_allow_html=True)

    # ── ベースライン状態を先に初期化（日付変更チェック含む）───
    init_baseline_state()

    # ── session_state 初期化 ─────────────────────────────────
    defaults = {
        "app_stage": "top",
        "phase_results": [],
        "current_phase": 0,
        "baseline_frames": [],
        "baseline_start": None,
        "phase_frames": [],
        "phase_start": None,
        "eq_scores": None,
        "eq_source_label": "",
        "session_saved": False,
        "debug_mode": False,
        "recorder": None,
        "saved_session_id": None,
        "baseline_saved_once": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── FaceLandmarker を取得（キャッシュ済みインスタンス）──
    face_landmarker = get_face_landmarker()

    # ================================================================
    # サイドバー: ページナビゲーション
    # ================================================================
    with st.sidebar:
        st.markdown("## 📍 ナビゲーション")
        page = st.radio(
            "ページを選択",
            ["🎥 計測", "📈 履歴", "🗣️ 会議モード"],   # ★NEW: 会議モード追加
            label_visibility="collapsed",
            key="nav_page",
        )
        st.markdown("---")
        st.session_state["debug_mode"] = st.checkbox(
            "🔍 デバッグモード",
            value=st.session_state.get("debug_mode", False),
            help="EQスコアの raw 値・比率テーブルを表示します",
        )

    # ── 履歴ページ ────────────────────────────────────────────
    if page == "📈 履歴":
        ui_history.render_history_page()
        _render_footer()
        return

    # ── 会議モードページ ★NEW ──────────────────────────────────
    if page == "🗣️ 会議モード":
        continuous_mode_ui.render_continuous_mode_page()
        _render_footer()
        return

    # ================================================================
    # 計測ページ（🎥 計測）
    # ================================================================
    # ── TOP ──────────────────────────────────────────────────
    if st.session_state.app_stage == "top":
        st.markdown("## 📋 ステップ1: ベースライン計測")
        st.markdown("""
        <div class="info-box">
            カメラに向かって自然な表情を保ってください。<br>
            Blendshapeデータを収集してベースラインを設定します。
        </div>""", unsafe_allow_html=True)

        if st.session_state.baseline:
            recorded_at = st.session_state.get("baseline_recorded_at", "不明")
            st.markdown(
                f'<div class="success-box">✅ 本日取得済みのベースラインがあります'
                f'（取得時刻: {recorded_at}）</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "🚀 本日のベースラインで計測開始",
                    type="primary", use_container_width=True,
                ):
                    st.session_state["recorder"] = DataRecorder()
                    st.session_state["saved_session_id"] = None
                    st.session_state.update({
                        "phase_results": [],
                        "current_phase": 0,
                        "phase_frames": [],
                        "phase_start": None,
                        "eq_scores": None,
                        "eq_source_label": "",
                        "session_saved": False,
                        "app_stage": "measure",
                    })
                    st.rerun()
            with c2:
                if st.button("🔄 ベースラインを再取得", use_container_width=True):
                    st.session_state.update({
                        "baseline_frames": [],
                        "baseline_start": None,
                        "baseline_saved_once": False,
                        "eq_scores": None,
                        "eq_source_label": "",
                        "session_saved": False,
                        "saved_session_id": None,
                        "app_stage": "baseline",
                        "force_manual_baseline": True,
                    })
                    st.rerun()
        else:
            st.markdown(
                '<div class="info-box">📌 本日のベースラインが未取得です。'
                '取得が必須です。</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "▶️ ベースライン取得を開始",
                type="primary", use_container_width=True,
            ):
                st.session_state.update({
                    "baseline_frames": [],
                    "baseline_start": None,
                    "baseline_saved_once": False,
                    "app_stage": "baseline",
                    "force_manual_baseline": False,
                })
                st.rerun()

    # ── BASELINE ─────────────────────────────────────────────
    elif st.session_state.app_stage == "baseline":
        st.markdown("## 📷 ステップ1: ベースライン計測")
        st.markdown("""
        <div class="info-box">
            😐 リラックスした状態で正面を向いてください<br>
            表情を作らず、自然な状態で <b>5秒間</b> 静止します。
        </div>""", unsafe_allow_html=True)

        if st.session_state.baseline_start is None:
            st.session_state.baseline_start = time.time()

        elapsed = time.time() - st.session_state.baseline_start

        if elapsed >= BASELINE_SEC:
            stop_camera_thread()

            if not st.session_state.get("baseline_saved_once", False):
                bs_frames: list = st.session_state.baseline_frames
                if bs_frames:
                    avg_frame = average_frames(bs_frames)
                    is_manual = st.session_state.pop("force_manual_baseline", False)
                    new_baseline_id = save_baseline_entry(
                        avg_frame.values, is_manual=is_manual
                    )
                    st.session_state.baseline = avg_frame
                    st.session_state.baseline_id = new_baseline_id
                    st.session_state.baseline_date = datetime.date.today().isoformat()
                    st.session_state.baseline_recorded_at = \
                        datetime.datetime.now().isoformat(timespec="seconds")
                    st.session_state["baseline_saved_once"] = True

                    st.markdown(
                        f'<div class="success-box">'
                        f'✅ ベースライン取得完了（{len(bs_frames)} フレーム平均）<br>'
                        f'baseline_id: <code>{new_baseline_id}</code>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("⚠️ 顔が検出できませんでした。再度お試しください。")
            else:
                saved_id = st.session_state.get("baseline_id", "")
                st.markdown(
                    f'<div class="success-box">'
                    f'✅ ベースライン取得完了<br>'
                    f'baseline_id: <code>{saved_id}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if st.button("▶️ 表情計測へ進む", type="primary"):
                st.session_state["recorder"] = DataRecorder()
                st.session_state["saved_session_id"] = None
                st.session_state.update({
                    "phase_results": [],
                    "current_phase": 0,
                    "phase_frames": [],
                    "phase_start": None,
                    "eq_scores": None,
                    "eq_source_label": "",
                    "session_saved": False,
                    "app_stage": "measure",
                })
                st.rerun()

        else:
            col_cam, col_info = st.columns([3, 1])
            with col_cam:
                frame_ph = st.empty()
            with col_info:
                prog_ph = st.empty()
                status_ph = st.empty()
                face_ph = st.empty()

            ct = get_camera_thread()
            frame = ct.get_frame()

            if frame is not None:
                bs_frame = get_blendshapes(frame, face_landmarker)
                if bs_frame is not None:
                    st.session_state.baseline_frames.append(bs_frame)
                    face_ph.markdown(
                        '<div class="success-box" style="padding:0.5rem;">'
                        '✅ 顔を検出しました</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    face_ph.markdown(
                        '<div class="info-box" style="padding:0.5rem;">'
                        '🔍 顔を探しています...</div>',
                        unsafe_allow_html=True,
                    )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_ph.image(frame_rgb, channels="RGB", width="stretch")

            prog_ph.progress(min(elapsed / BASELINE_SEC, 1.0))
            status_ph.caption(
                f"収集フレーム: {len(st.session_state.baseline_frames)}\n"
                f"経過: {elapsed:.1f}s / {BASELINE_SEC}s"
            )
            time.sleep(0.05)
            st.rerun()

    # ── MEASURE ──────────────────────────────────────────────
    elif st.session_state.app_stage == "measure":
        phase_idx = st.session_state.current_phase
        phase = PHASES[phase_idx]
        tip = phase.get("tip", "")
        done = len(st.session_state.phase_results) > phase_idx

        st.markdown(
            f"## 🎭 ステップ2: 表情計測 ({phase_idx + 1}/{len(PHASES)})"
        )
        st.progress(phase_idx / len(PHASES))

        hint_html = f"<br>💡 <i>{tip}</i>" if tip else ""
        st.markdown(f"""
        <div class="info-box">
            <b>{phase['label']}</b><br>
            👉 {phase['instruction']}<br>
            約 <b>{PHASE_SEC}秒間</b> その表情を維持してください。{hint_html}
        </div>""", unsafe_allow_html=True)

        if done:
            stop_camera_thread()
            st.markdown(
                '<div class="success-box">✅ このフェーズの計測が完了しました</div>',
                unsafe_allow_html=True,
            )
            next_idx = phase_idx + 1
            if next_idx < len(PHASES):
                if st.button(
                    f"▶️ 次へ: {PHASES[next_idx]['label']}", type="primary",
                ):
                    st.session_state.current_phase = next_idx
                    st.session_state.phase_start = None
                    st.session_state.phase_frames = []
                    st.rerun()
            else:
                if st.button("📊 スコアを見る", type="primary"):
                    save_scores(st.session_state.phase_results)
                    st.session_state["eq_scores"] = None
                    st.session_state["eq_source_label"] = ""
                    st.session_state["session_saved"] = False
                    st.session_state.app_stage = "result"
                    st.rerun()

        else:
            if st.session_state.phase_start is None:
                st.session_state.phase_start = time.time()
                st.session_state.phase_frames = []

            elapsed = time.time() - st.session_state.phase_start

            if elapsed >= PHASE_SEC:
                frames: list = st.session_state.phase_frames
                if frames:
                    result = {
                        "phase": phase["key"],
                        "tension":  float(np.mean([s.tension for s in frames])),
                        "anger":    float(np.mean([s.anger for s in frames])),
                        "positive": float(np.mean([s.positive for s in frames])),
                        "stable":   float(np.mean([s.stable for s in frames])),
                    }
                    axes_dict = {k: result[k] for k in AXES}
                    main_dir = max(axes_dict, key=axes_dict.get)
                    result["dominant"] = (
                        main_dir if axes_dict[main_dir] >= 0.05 else "none"
                    )
                else:
                    result = {
                        "phase": phase["key"],
                        "tension": 0.0,
                        "anger": 0.0,
                        "positive": 0.0,
                        "stable": 1.0,
                        "dominant": "stable",
                    }

                st.session_state.phase_results.append(result)
                st.session_state.phase_start = None
                st.session_state.phase_frames = []
                st.rerun()

            else:
                col_cam, col_info = st.columns([3, 1])
                with col_cam:
                    frame_ph = st.empty()
                with col_info:
                    prog_ph = st.empty()
                    status_ph = st.empty()
                    score_ph = st.empty()

                ct = get_camera_thread()
                frame = ct.get_frame()

                if frame is not None:
                    bs_frame = get_blendshapes(frame, face_landmarker)
                    if bs_frame is not None:
                        dir_scores: DirectionScores = calc_direction_scores(
                            bs_frame,
                            st.session_state.baseline,
                        )
                        st.session_state.phase_frames.append(dir_scores)

                        recorder: "DataRecorder | None" = st.session_state.get("recorder")
                        if recorder is not None:
                            recorder.record_frame(
                                blendshapes=bs_frame.values,
                                phase=phase["key"],
                            )

                        frame = draw_overlay(frame, dir_scores)
                        score_ph.markdown(f"""
                        <div style="
                            background: #161b22;
                            border: 1px solid #30363d;
                            border-radius: 8px;
                            padding: 0.8rem;
                            font-size: 0.85rem;
                        ">
                            😰 緊張: {dir_scores.tension * 100:.0f}%<br>
                            😠 怒り: {dir_scores.anger * 100:.0f}%<br>
                            😊 ポジ: {dir_scores.positive * 100:.0f}%<br>
                            😌 安定: {dir_scores.stable * 100:.0f}%<br>
                            <hr style="border-color:#30363d;margin:0.4rem 0;">
                            <b>主方向: {dir_scores.dominant()}</b>
                        </div>""", unsafe_allow_html=True)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_ph.image(frame_rgb, channels="RGB",
                                   use_container_width=True)

                prog_ph.progress(min(elapsed / PHASE_SEC, 1.0))
                status_ph.caption(
                    f"⏱️ {elapsed:.1f} / {PHASE_SEC}s\n"
                    f"検出: {len(st.session_state.phase_frames)}フレーム"
                )
                time.sleep(0.05)
                st.rerun()

    # ── RESULT ───────────────────────────────────────────────
    elif st.session_state.app_stage == "result":
        render_results(st.session_state.phase_results)

    # ── フッター ─────────────────────────────────────────────
    _render_footer()


# ============================================================
# フッター（共通）
# ============================================================
def _render_footer():
    st.markdown(f"""
    <div style="
        text-align: center;
        color: #484f58;
        font-size: 0.8rem;
        margin-top: 3rem;
        border-top: 1px solid #21262d;
        padding-top: 1rem;
    ">
        🧠 EQ GYM | Powered by MediaPipe FaceLandmarker (52 blendshapes)<br>
        📅 {datetime.date.today().strftime('%Y年%m月%d日')} |
        🔒 カメラ映像はローカル処理のみ
    </div>""", unsafe_allow_html=True)


# ============================================================
# エントリポイント
# ============================================================
if __name__ == "__main__":
    main()