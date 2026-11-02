# ============================================================
# baseline_store.py
# 役割：日次ベースラインの保存・取得を管理する
#
# 仕様（確定版）:
# - 日付が変わったらベースライン再取得は必須（自動フォールバックしない）
# - 同日内であれば複数回のベースライン取得を許可
#   （is_manual=True で手動再取得を区別）
# - baseline_id (UUID) を発行し、sessions.csv から参照させる
# - 同日内で複数存在する場合は recorded_at が最新のものを採用する
# - 保存先: data/aggregated/baseline_scores.csv
#
# 【重要】date列の型ゆれ対策（2026-09-14 障害対応）:
# - Excel等でCSVを直接開いて保存すると、"2026-09-14" のような
#   日付風文字列が自動的にシリアル値（例: 46279）に変換されてしまう
#   事故が発生しうる。
# - get_today_latest_baseline() では date列の値が
#     ① "2026-09-14" のような文字列
#     ② "2026-09-14T10:00:00" のようなISO8601文字列
#     ③ 46279 のようなExcelシリアル値（int/float、または数字だけの文字列）
#   のいずれであっても正しく本日の日付として判定できるように
#   正規化処理 (_normalize_date) を経由してから比較する。
# - 恒久対策として、date列に依存せず recorded_at から日付を都度
#   導出する設計への移行も検討可（本ファイルでは互換性を優先し
#   date列はそのまま維持しつつ防御的に正規化する方式を採用）。
# ============================================================
import uuid
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from data_recorder import AGGREGATED_DIR  # 既存のディレクトリ構成を再利用

BASELINE_CSV_PATH = AGGREGATED_DIR / "baseline_scores.csv"
LEGACY_ROOT_BASELINE_CSV = Path("baseline_scores.csv")  # 旧実装（単一行上書き）

# メタデータ列（Blendshape特徴量列と区別するため）
META_COLUMNS = ["baseline_id", "date", "recorded_at", "is_manual"]

# Excelの日付シリアル値の基準日（1900年うるう年バグを踏まえた実務上の定数）
_EXCEL_EPOCH = datetime(1899, 12, 30)


def load_baseline_scores() -> pd.DataFrame:
    """baseline_scores.csv を DataFrame として読み込む。存在しなければ空DataFrame。"""
    if not BASELINE_CSV_PATH.exists():
        return pd.DataFrame(columns=META_COLUMNS)
    return pd.read_csv(BASELINE_CSV_PATH)


def _normalize_date(value) -> str:
    """
    date列の値を "YYYY-MM-DD" 形式の文字列に正規化する。

    対応するケース:
    - "2026-09-14"                      → そのまま先頭10文字を使用
    - "2026-09-14T23:35:08"              → ISO8601文字列の先頭10文字を使用
    - 46279 (int/float)                  → Excelシリアル値として日付変換
    - "46279" (数字だけの文字列)          → 同上
    - NaN / None / 変換不能な値          → 空文字列を返す（本日とは絶対に一致しない）

    Parameters
    ----------
    value : Any
        DataFrame の date列から取り出した1セルの値

    Returns
    -------
    str : "YYYY-MM-DD" 形式の文字列、または正規化不能時は ""
    """
    if value is None:
        return ""

    # NaN（float）を検出
    if isinstance(value, float) and pd.isna(value):
        return ""

    # ケース1: 数値型（Excelシリアル値の疑い）
    if isinstance(value, (int, float)):
        try:
            return (_EXCEL_EPOCH + timedelta(days=float(value))).strftime("%Y-%m-%d")
        except (OverflowError, ValueError):
            return ""

    # ケース2: 文字列型
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return ""

        # 数字だけの文字列 → Excelシリアル値として扱う
        if s.isdigit():
            try:
                return (_EXCEL_EPOCH + timedelta(days=float(s))).strftime("%Y-%m-%d")
            except (OverflowError, ValueError):
                return ""

        # "2026-09-14" または "2026-09-14T23:35:08" 等 → 先頭10文字を日付とみなす
        candidate = s[:10]
        try:
            # 妥当な日付かどうかを検証（不正な文字列を誤って通さないため）
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            return ""

    # その他の型は非対応
    return ""


def get_today_latest_baseline() -> tuple[str | None, dict | None, str | None]:
    """
    本日の日付に紐づく最新のベースラインを取得する。

    date列がExcel等によってシリアル値に破損しているケースにも対応するため、
    比較前に _normalize_date() で正規化を行う。

    Returns
    -------
    (baseline_id, feature_dict, recorded_at)
        本日分が存在しない場合は (None, None, None)。
        feature_dict にはメタデータ列を除いた Blendshape 特徴量のみを含む。
    """
    df = load_baseline_scores()
    if df.empty:
        return None, None, None

    # date列を正規化した一時列を作成して比較する
    df = df.copy()
    df["_date_norm"] = df["date"].apply(_normalize_date)

    today = datetime.now().strftime("%Y-%m-%d")
    matched = df[df["_date_norm"] == today]
    if matched.empty:
        return None, None, None

    latest_row = matched.sort_values("recorded_at").iloc[-1]
    baseline_id = latest_row["baseline_id"]
    recorded_at = latest_row["recorded_at"]
    feature_dict = {
        k: float(v)
        for k, v in latest_row.items()
        if k not in META_COLUMNS and k != "_date_norm"
    }
    return baseline_id, feature_dict, recorded_at


def save_baseline_entry(features: dict, is_manual: bool = False) -> str:
    """
    新しいベースラインを1行追記する（既存データは上書きしない）。

    Parameters
    ----------
    features : dict
        {blendshape_name: value, ...} 52特徴量（BlendshapeFrame.values）
    is_manual : bool
        手動再取得によるものかどうか

    Returns
    -------
    str : 新規発行された baseline_id
    """
    now = datetime.now()
    baseline_id = f"base-{uuid.uuid4().hex[:8]}"

    row = {
        "baseline_id": baseline_id,
        "date": now.strftime("%Y-%m-%d"),
        "recorded_at": now.isoformat(timespec="seconds"),
        "is_manual": is_manual,
        **features,
    }

    df = load_baseline_scores()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    BASELINE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASELINE_CSV_PATH, index=False)

    return baseline_id


def _migrate_legacy_root_baseline():
    """
    旧実装（プロジェクトルートの baseline_scores.csv、単一行上書き）が
    存在し、新形式がまだ存在しない場合、1件のレコードとして移行する。
    ※移行後の date はファイルの最終更新日時になるため、
      本日日付と一致しなければ「本日分なし」として再取得が必須になる
      （仕様通りの挙動）。
    """
    if not LEGACY_ROOT_BASELINE_CSV.exists():
        return
    if BASELINE_CSV_PATH.exists():
        return  # 新形式が既に存在する場合は何もしない

    try:
        df_legacy = pd.read_csv(LEGACY_ROOT_BASELINE_CSV)
    except Exception:
        return
    if df_legacy.empty:
        return

    features = df_legacy.iloc[0].to_dict()
    mtime = datetime.fromtimestamp(LEGACY_ROOT_BASELINE_CSV.stat().st_mtime)

    row = {
        "baseline_id": f"base-{uuid.uuid4().hex[:8]}",
        "date": mtime.strftime("%Y-%m-%d"),
        "recorded_at": mtime.isoformat(timespec="seconds"),
        "is_manual": False,
        **{k: float(v) for k, v in features.items()},
    }

    BASELINE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(BASELINE_CSV_PATH, index=False)


# モジュール読み込み時に一度だけ移行処理を実行
_migrate_legacy_root_baseline()