# app.py ── Streamlit (https://<YOUR-APP>.streamlit.app)

import io
from pathlib import Path
from datetime import date, timedelta
from pandas.io.formats.style import Styler

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import re


# ✅ Ag-Grid (합계행 상단 고정 & 컬럼 필터/정렬/선택 지원) ── (사용 안 해도 됨: 데이터 매트릭스는 st.data_editor 사용)
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
except Exception as _e:
    AgGrid = None

# ================= 기본 설정 =================
st.set_page_config(page_title="외부요인 기반 빅데이터 철도 수요예측 플랫폼", layout="wide")

# ======= 상단 타이틀 + 다크모드 토글 =======
title_col, theme_col = st.columns([1,0.18])
with title_col:
    st.title("📈 외부요인 기반 빅데이터 철도 수요예측 플랫폼")
with theme_col:
    DARK = st.checkbox("🌙 다크 모드", value=False)

# ---------- 테마 색상 변수 ----------
if DARK:
    BG        = "#0B1220"; SURFACE="#111827"; PANEL_BG="#0F172A"; BORDER="#1F2937"
    TEXT      = "#E5E7EB"; SUBTEXT="#9CA3AF"; SHADOW="rgba(0,0,0,0.35)"
    HILITE1   = "rgba(56,189,248,0.12)"; HILITE2="rgba(148,163,184,0.10)"
    CARD_BG   = "#111827"; CARD_BORDER="#374151"; PLOTLY_TEMPLATE="plotly_dark"
    SUM_BG    = "rgba(56,189,248,0.10)"
else:
    BG        = "#FFFFFF"; SURFACE="#FFFFFF"; PANEL_BG="#F8FAFC"; BORDER="#E5E7EB"
    TEXT      = "#111827"; SUBTEXT="#6B7280"; SHADOW="rgba(0,0,0,0.05)"
    HILITE1   = "rgba(30,144,255,0.08)"; HILITE2="rgba(100,116,139,0.06)"
    CARD_BG   = "#FFFFFF"; CARD_BORDER="#E5E7EB"; PLOTLY_TEMPLATE="plotly_white"
    SUM_BG    = "rgba(30,144,255,0.08)"

# ---- 글로벌 스타일 ----
st.markdown(
    f"""
    <style>
    html, body, .stApp {{ background: {BG}; color:{TEXT}; }}
    .panel {{
        border: 1px solid {BORDER};
        background: {PANEL_BG};
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 4px 18px {SHADOW};
        margin-bottom: 12px;
    }}
    .lg-line   {{ height:2px; border-top: 4px solid #1f77b4; border-radius:2px; display:inline-block; width:22px; margin-right:6px; }}
    .lg-line-dash {{
        height:0; display:inline-block; width:22px; margin-right:6px; position:relative; top:2px;
        border-top: 0; background: repeating-linear-gradient(90deg, #1f77b4 0 6px, rgba(0,0,0,0) 6px 12px);
    }}
    .lg-bar    {{ background:#ff7f0e; display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; }}
    .lg-bar-f  {{ background:#ff7f0e; opacity:0.7; display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; }}
    .lg-text   {{ font-size: 13px; color:{TEXT}; vertical-align:middle; }}
    .legend-row {{ display:flex; gap:18px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }}
    div[data-testid="stDataFrame"] div[role="grid"] {{ background: {SURFACE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== 기간 정의 =====
ACT_START = pd.to_datetime("2020-08-01")
ACT_END   = pd.to_datetime("2025-08-31")  # ✅ 실적 종료
FCT_START = pd.to_datetime("2025-09-01")
FCT_END   = pd.to_datetime("2025-11-29")

# ================= 파일 로더 =================
@st.cache_data(show_spinner=False)
def load_df_from_repo_csv(filename: str):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"'{filename}' 파일을 찾을 수 없습니다.")
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

# ================= 휴일 로더 =================
@st.cache_data(show_spinner=False)
def load_holidays_df() -> pd.DataFrame:
    try:
        df = load_df_from_repo_csv("holidays_rows.csv").copy()
    except FileNotFoundError:
        st.warning("'holidays_rows.csv' 파일을 찾지 못했습니다. 휴무 표시는 생략됩니다.")
        return pd.DataFrame(columns=["holiday_date","name"])
    if "holiday_date" not in df.columns:
        st.warning("'holidays_rows.csv'에 'holiday_date' 컬럼이 없습니다. 휴무 표시는 생략됩니다.")
        return pd.DataFrame(columns=["holiday_date","name"])
    if "name" not in df.columns:
        df["name"] = ""
    df["holiday_date"] = pd.to_datetime(df["holiday_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["holiday_date"]).drop_duplicates(subset=["holiday_date"]).sort_values("holiday_date")
    return df[["holiday_date","name"]]

# ================= 실적 / 예측 로더 =================
@st.cache_data(show_spinner=False)
def load_actual_df() -> pd.DataFrame:
    df = load_df_from_repo_csv("merged.csv").copy()
    cols = {c.lower(): c for c in df.columns}
    df.rename(columns={cols.get("date","date"):"date",
                       cols.get("passengers","passengers"):"passengers",
                       cols.get("sales_amount","sales_amount"):"sales_amount"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[(df["date"] >= ACT_START) & (df["date"] <= ACT_END)]
    df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
    df["sales_amount"] = pd.to_numeric(df["sales_amount"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_forecast_df() -> pd.DataFrame:
    f = load_df_from_repo_csv("forecast_pass.csv").copy()
    cols = {c.lower(): c for c in f.columns}
    f.rename(columns={cols.get("date","date"):"date",
                      cols.get("forecast_90d","forecast_90d"):"passengers"}, inplace=True)
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f.dropna(subset=["date"]).sort_values("date")
    f = f[(f["date"] >= FCT_START) & (f["date"] <= FCT_END)]
    f["passengers"] = pd.to_numeric(f["passengers"], errors="coerce")
    f["sales_amount"] = np.nan
    return f

@st.cache_data(show_spinner=False)
def load_forecast_sales_df() -> pd.DataFrame:
    f = load_df_from_repo_csv("forecast_sales.csv").copy()
    cols = {c.lower(): c for c in f.columns}
    f.rename(columns={cols.get("date","date"):"date",
                      cols.get("forecast_90d","forecast_90d"):"pred_sales_amount"}, inplace=True)
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f.dropna(subset=["date"]).sort_values("date")
    f = f[(f["date"] >= FCT_START) & (f["date"] <= FCT_END)]
    f["pred_sales_amount"] = pd.to_numeric(f["pred_sales_amount"], errors="coerce")
    return f[["date","pred_sales_amount"]]

@st.cache_data(show_spinner=False)
def load_actual_rows_df() -> pd.DataFrame:
    df = load_df_from_repo_csv("train_reservations_rows.csv").copy()
    required = ["travel_date","passengers","sales_amount"]
    miss = [c for c in required if c not in df.columns]
    if miss: raise KeyError(f"'train_reservations_rows.csv'에 다음 컬럼 필요: {', '.join(miss)}")
    df["travel_date"] = pd.to_datetime(df["travel_date"], errors="coerce")
    df = df.dropna(subset=["travel_date"])
    for c in ["passengers","sales_amount"]:
        df[c] = df[c].astype(str).str.replace(",", "", regex=False).replace("nan", np.nan)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    daily = (df.assign(date=df["travel_date"].dt.floor("D"))
               .groupby("date", as_index=False)[["passengers","sales_amount"]].sum()
               .sort_values("date"))
    daily = daily[(daily["date"] >= ACT_START) & (daily["date"] <= ACT_END)]
    return daily

# ================= 유틸 =================
KO_DAYS = ["월","화","수","목","금","토","일"]
def fmt_date_ko(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m-%d") + " (" + s.dt.weekday.map(dict(enumerate(KO_DAYS))) + ")"

def ensure_in_range(s: pd.Timestamp, e: pd.Timestamp, lo: pd.Timestamp, hi: pd.Timestamp):
    s2 = max(s, lo); e2 = min(e, hi)
    if s2 > e2: s2, e2 = lo, lo
    return s2, e2

def align_last_year_same_weekday(r_s: pd.Timestamp, n_days: int):
    raw = (r_s - pd.DateOffset(years=1)).normalize()
    diff = (r_s.weekday() - raw.weekday()) % 7
    l_s = raw + pd.Timedelta(days=diff)
    l_e = l_s + pd.Timedelta(days=n_days-1)
    l_s, l_e = ensure_in_range(l_s, l_e, ACT_START, ACT_END)
    cur = (l_e - l_s).days + 1
    if cur < n_days:
        deficit = n_days - cur
        l_s = max(ACT_START, l_s - pd.Timedelta(days=deficit))
        l_e = l_s + pd.Timedelta(days=n_days-1)
    return ensure_in_range(l_s, l_e, ACT_START, ACT_END)

def force_same_length(left_s, left_e, right_s, right_e):
    n = (right_e - right_s).days + 1
    left_e_target = left_s + pd.Timedelta(days=n-1)
    left_s2, left_e2 = ensure_in_range(left_s, left_e_target, ACT_START, ACT_END)
    cur = (left_e2 - left_s2).days + 1
    if cur < n:
        deficit = n - cur
        left_s2 = max(ACT_START, left_s2 - pd.Timedelta(days=deficit))
    left_s2, left_e2 = ensure_in_range(left_s2, left_s2 + pd.Timedelta(days=n-1), ACT_START, ACT_END)
    right_s2, right_e2 = ensure_in_range(right_s, right_e, FCT_START, FCT_END)
    return left_s2, left_e2, right_s2, right_e2

# --------- 외부요인(이벤트 합계) + 휴일 플래그 ----------
@st.cache_data(show_spinner=False)
def load_external_factors_df() -> pd.DataFrame:
    try:
        df = load_df_from_repo_csv("merged.csv").copy()
    except FileNotFoundError:
        st.warning("'merged.csv' 파일을 찾지 못해 이벤트 합계를 표시할 수 없습니다.")
        return pd.DataFrame(columns=["date","event_sum"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        st.warning("'merged.csv'에 'date' 컬럼이 없습니다.")
        return pd.DataFrame(columns=["date","event_sum"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    event_cols = ["bexco_events_count","coex_events_count","kintex_events_count",
                  "games_baseball","games_soccer","concerts_events_count"]
    for c in event_cols:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["event_sum"] = df[event_cols].sum(axis=1)
    df = df[(df["date"] >= ACT_START) & (df["date"] <= ACT_END)]
    return df[["date","event_sum"]]

def build_event_strings(dates_index: pd.DatetimeIndex, factors_df: pd.DataFrame) -> list[str]:
    if dates_index is None or len(dates_index) == 0: return []
    if isinstance(factors_df, pd.DataFrame) and not factors_df.empty and "date" in factors_df.columns:
        f = factors_df.set_index("date").reindex(dates_index)
        ev = pd.to_numeric(f.get("event_sum"), errors="coerce").fillna(0)
    else:
        ev = pd.Series([0]*len(dates_index), index=dates_index)
    return [f"이벤트 {int(round(v))}" for v in ev]

# === 휴일명 6자 라벨 + 툴팁 전체명 ===
def _truncate_with_ellipsis(text: str, max_len: int = 6) -> str:
    if not isinstance(text, str):
        return ""
    return text if len(text) <= max_len else (text[:max_len] + "…")

def build_holiday_labels(dates_index: pd.DatetimeIndex, holidays_df: pd.DataFrame, max_len: int = 6):
    if dates_index is None or len(dates_index) == 0:
        return [], []
    if holidays_df is None or holidays_df.empty or "holiday_date" not in holidays_df.columns:
        return ["" for _ in range(len(dates_index))], ["" for _ in range(len(dates_index))]
    hmap = holidays_df.set_index("holiday_date")["name"]
    labels, fulls = [], []
    for d in dates_index:
        name = hmap.get(d.normalize(), "")
        if isinstance(name, float) and np.isnan(name):
            name = ""
        if name:
            labels.append(_truncate_with_ellipsis(str(name), max_len))
            fulls.append(str(name))
        else:
            labels.append("")
            fulls.append("")
    return labels, fulls

import streamlit as st

def set_sidebar_font(size_px: int = 16, label_px: int | None = None, line_height: float = 1.35):
    """
    Streamlit 사이드바 폰트 크기/줄간격을 CSS로 일괄 조정합니다.
    - size_px: 사이드바 기본 폰트 크기(px)
    - label_px: 위젯 라벨(예: radio, date_input 라벨) 크기(px). None이면 size_px 사용
    - line_height: 줄간격
    """
    if label_px is None:
        label_px = size_px
    st.markdown(
        f"""
        <style>
        /* 사이드바 영역 전체 */
        [data-testid="stSidebar"] * {{
            font-size: {size_px}px !important;
            line-height: {line_height} !important;
        }}
        /* 섹션 헤더/서브헤더(크게 보이게 약간 증폭) */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            font-size: {int(size_px*1.15)}px !important;
            font-weight: 700 !important;
        }}
        /* 위젯 라벨(예: radio, date_input 레이블) */
        [data-testid="stSidebar"] label,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {{
            font-size: {label_px}px !important;
            font-weight: 600 !important;
        }}
        /* date_input 필드 내부 글자 */
        [data-testid="stSidebar"] [data-testid="stDateInput"] input {{
            font-size: {size_px}px !important;
        }}
        /* radio 항목 라벨 */
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {{
            font-size: {size_px}px !important;
        }}
        /* 구분선 여백 살짝 넉넉하게 */
        [data-testid="stSidebar"] hr {{
            margin: 0.6rem 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

import streamlit as st

def set_sidebar_style(font_size=16, line_height=1.6, paragraph_gap="0.8rem"):
    """
    Streamlit 사이드바 폰트 및 문단 간격 스타일 설정
    - font_size: 기본 폰트 크기(px)
    - line_height: 줄 간격(line-height)
    - paragraph_gap: 문단(p, div 등) 사이 여백 (rem 또는 px)
    """
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] * {{
            font-size: {font_size}px !important;
            line-height: {line_height} !important;
        }}
        /* 문단 간 간격 */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] label {{
            margin-bottom: {paragraph_gap} !important;
        }}
        /* 위젯 간 여백도 넉넉하게 */
        [data-testid="stSidebar"] .stRadio,
        [data-testid="stSidebar"] .stDateInput,
        [data-testid="stSidebar"] .stSelectbox {{
            margin-bottom: {paragraph_gap} !important;
        }}
        /* 구분선(hr) 상하 여백 */
        [data-testid="stSidebar"] hr {{
            margin: 1rem 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ===================== 스타일: 사이드바 폰트 =====================
set_sidebar_font(size_px=20, label_px=18, line_height=1.4)

# ===================== 사이드바: 기간 선택 =====================
# ===== 스타일 설정 =====
set_sidebar_style(font_size=17, line_height=1.6, paragraph_gap="0.6rem")

# ===== 사이드바 예시 =====
st.sidebar.markdown("---")
st.sidebar.subheader("📅 기간 선택")

default_right_start = date(2025, 9, 1)
default_right_end   = date(2025, 9, 7)
right_range = st.session_state.get("right_range", (default_right_start, default_right_end))
right_sel = st.sidebar.date_input(
    "① 예측 기간 (YYYY-MM-DD)",
    value=right_range, min_value=FCT_START.date(), max_value=FCT_END.date(), key="right_picker_sidebar"
)

# ── (교체) 실적 기간 모드: 가로 라디오, (O) 느낌
mode_options = ["사용 안함 (예측만)", "전년도 동일(일자)", "전년도 동일(요일)", "사용자 지정"]
st.sidebar.markdown(
    """
    <style>
    /* 사이드바 라디오를 가로로 보기 좋게 */
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 10px !important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] { margin-right: 8px !important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label p { font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True
)
left_mode = st.sidebar.radio(
    "② 실적 기간 모드",
    options=mode_options,
    index=1,
    key="left_mode_sidebar",
    horizontal=True,  # ← 가로 배치
)


if left_mode == "사용자 지정":
    left_range = st.session_state.get("left_range", (date(2024, 9, 1), date(2024, 9, 7)))
    left_sel = st.sidebar.date_input(
        "실적 기간 (YYYY-MM-DD)",
        value=left_range, min_value=ACT_START.date(), max_value=ACT_END.date(), key="left_picker_sidebar"
    )


# ================= 기간 정규화/동기화 =================
def norm_tuple(sel):
    return sel if isinstance(sel, tuple) else (sel, sel)

r_s, r_e = map(pd.to_datetime, norm_tuple(right_sel))
r_s, r_e = ensure_in_range(r_s, r_e, FCT_START, FCT_END)
N_days = (r_e - r_s).days + 1

if left_mode == "사용 안함 (예측만)":
    l_s, l_e = None, None
elif left_mode == "전년도 동일(일자)":
    l_s = (r_s - pd.DateOffset(years=1)).normalize(); l_e = l_s + pd.Timedelta(days=N_days-1)
    l_s, l_e = ensure_in_range(l_s, l_e, ACT_START, ACT_END)
elif left_mode == "전년도 동일(요일)":
    l_s, l_e = align_last_year_same_weekday(r_s, N_days)
else:
    l_s, l_e = map(pd.to_datetime, norm_tuple(left_sel))
    l_s, l_e, r_s, r_e = force_same_length(l_s, l_e, r_s, r_e)

st.session_state["right_range"] = (r_s.date(), r_e.date())
if left_mode == "사용자 지정" and l_s is not None:
    st.session_state["left_range"] = (l_s.date(), l_e.date())

# ================= 외부 데이터 로드 =================
actual_df_all   = load_actual_df()
forecast_df_all = load_forecast_df()
external_factors_df = load_external_factors_df()
holidays_df = load_holidays_df()

try:
    forecast_sales_all = load_forecast_sales_df()
except FileNotFoundError as e:
    st.warning(f"예측 매출 파일을 찾지 못했습니다: {e}")
    forecast_sales_all = pd.DataFrame(columns=["date","pred_sales_amount"])

def get_range(df, s, e, tag):
    if s is None or e is None: return pd.DataFrame(columns=["date","passengers","sales_amount","source"])
    out = df[(df["date"] >= s) & (df["date"] <= e)].copy(); out["source"] = tag; return out

left_df  = get_range(actual_df_all,   l_s, l_e, "actual") if l_s is not None else pd.DataFrame(columns=["date","passengers","sales_amount","source"])
right_df = get_range(forecast_df_all, r_s, r_e, "forecast")

if not right_df.empty:
    right_df = right_df.merge(forecast_sales_all, on="date", how="left")
    right_df["sales_amount"] = np.where(right_df["sales_amount"].isna(), right_df["pred_sales_amount"], right_df["sales_amount"])

df_sel = pd.concat(
    ([left_df.assign(period="실적기간")] if not left_df.empty else []) + [right_df.assign(period="예측기간")],
    ignore_index=True).sort_values("date") if (not right_df.empty or not left_df.empty) else pd.DataFrame(columns=["date","passengers","sales_amount","source","period"])

df_sel["sales_million"] = pd.to_numeric(df_sel["sales_amount"], errors="coerce")/1_000_000
df_sel["passengers_k"]  = pd.to_numeric(df_sel["passengers"], errors="coerce")/1_000

# ================= X축 카테고리 =================
order_left  = pd.date_range(l_s, l_e, freq="D") if l_s is not None else pd.DatetimeIndex([])
order_right = pd.date_range(r_s, r_e, freq="D")
category_array = (
    [f"실적|{d.strftime('%Y-%m-%d')}" for d in order_left] +
    [f"예측|{d.strftime('%Y-%m-%d')}" for d in order_right]
)
if not df_sel.empty:
    df_sel["x_cat"] = df_sel.apply(lambda r: f"{'실적' if r['period']=='실적기간' else '예측'}|{r['date'].strftime('%Y-%m-%d')}", axis=1)

# =================== 그래프 패널(분리) ===================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("📊그래프")

sp, cSales, cPax = st.columns([8,1.6,1.6])
with cSales: show_sales = st.checkbox("매출액", True, key="cb_sales")
with cPax:   show_pax   = st.checkbox("승객수", True, key="cb_pax")

def _add_watermark(fig, text: str):
    # 투명도 있는 워터마크 (레이어: below)
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=text, showarrow=False,
        font=dict(size=48, color="rgba(0,0,0,0.08)"),
        align="center", opacity=1.0
    )
    # 배경/플롯 색상 일치 & 그리드 보이되 은은하게
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),
        margin=dict(t=24, r=30, b=60, l=70),
        showlegend=False,
        font=dict(family="Nanum Gothic, Malgun Gothic, AppleGothic, Noto Sans KR, Sans-Serif", size=13, color=TEXT),
    )

def _build_single_fig(df: pd.DataFrame, title_text: str):
    fig = go.Figure()
    if df.empty:
        _add_watermark(fig, title_text)
        return fig

    # 승객수(막대, y2)
    if show_pax and ("passengers_k" in df.columns):
        fig.add_trace(go.Bar(
            x=df["date"], y=df["passengers_k"], name="승객수",
            marker=dict(line=dict(width=0)),
            opacity=0.55, yaxis="y2",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>승객수: %{y:,.0f} 천명<extra></extra>"
        ))
    # 매출(선, y1)
    if show_sales and ("sales_million" in df.columns):
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["sales_million"], name="매출액", mode="lines+markers",
            line=dict(width=2.6), marker=dict(size=6),
            yaxis="y1", connectgaps=True,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>매출액: %{y:,.0f} 백만원<extra></extra>"
        ))

    # 듀얼축
    fig.update_layout(
        xaxis=dict(title="", type="date", tickformat="%Y-%m-%d", tickangle=-45),
        yaxis=dict(title="매출액(백만원)", tickformat=",.0f"),
        yaxis2=dict(title="승객수(천명)", overlaying="y", side="right", tickformat=",.0f"),
        barmode="group", bargap=0.15, bargroupgap=0.05,
    )

    _add_watermark(fig, title_text)
    return fig

# 왼쪽/오른쪽 데이터 준비
left_plot_df  = pd.DataFrame()
right_plot_df = pd.DataFrame()
if not left_df.empty:
    left_plot_df = left_df.copy()
    left_plot_df["sales_million"] = pd.to_numeric(left_plot_df["sales_amount"], errors="coerce")/1_000_000
    left_plot_df["passengers_k"]  = pd.to_numeric(left_plot_df["passengers"],   errors="coerce")/1_000
if not right_df.empty:
    right_plot_df = right_df.copy()
    right_plot_df["sales_million"] = pd.to_numeric(right_plot_df["sales_amount"], errors="coerce")/1_000_000
    right_plot_df["passengers_k"]  = pd.to_numeric(right_plot_df["passengers"],   errors="coerce")/1_000

# 레이아웃: 실적이 없으면 예측이 전폭 사용
if left_plot_df.empty:
    fig_right = _build_single_fig(right_plot_df, "예측")
    st.plotly_chart(
        fig_right, use_container_width=True,
        config=dict(displaylogo=False,
                    toImageButtonOptions=dict(format="png", filename=f"forecast_{date.today()}", scale=2),
                    modeBarButtonsToAdd=["hovercompare"])
    )
else:
    colL, colR = st.columns(2)
    with colL:
        st.markdown("**✅실적**")
        fig_left = _build_single_fig(left_plot_df, "실적")
        st.plotly_chart(
            fig_left, use_container_width=True,
            config=dict(displaylogo=False,
                        toImageButtonOptions=dict(format="png", filename=f"actual_{date.today()}", scale=2),
                        modeBarButtonsToAdd=["hovercompare"])
        )
    with colR:
        st.markdown("**✅예측**")
        fig_right = _build_single_fig(right_plot_df, "예측")
        st.plotly_chart(
            fig_right, use_container_width=True,
            config=dict(displaylogo=False,
                        toImageButtonOptions=dict(format="png", filename=f"forecast_{date.today()}", scale=2),
                        modeBarButtonsToAdd=["hovercompare"])
        )

st.markdown('</div>', unsafe_allow_html=True)

# ===================== 표(실적/예측) =====================
left_dates  = pd.date_range(l_s, l_e, freq="D") if l_s is not None else pd.DatetimeIndex([])
right_dates = pd.date_range(r_s, r_e, freq="D")

left_tbl  = pd.DataFrame({"pos": range(len(left_dates)), "date": left_dates}) if len(left_dates)>0 else pd.DataFrame(columns=["pos","date"])
right_tbl = pd.DataFrame({"pos": range(len(right_dates)),"date": right_dates})

# 외부요인: 실적(이벤트/휴무여부 분리), 예측(휴무여부)
left_event_strings  = build_event_strings(left_dates, external_factors_df) if len(left_dates)>0 else []

if not left_df.empty:
    l_vals = left_df.set_index("date").reindex(left_dates)
    left_tbl["sales_million"] = (pd.to_numeric(l_vals["sales_amount"], errors="coerce")/1_000_000).values
    left_tbl["passengers_k"]  = (pd.to_numeric(l_vals["passengers"],   errors="coerce")/1_000).values
r_vals = right_df.set_index("date").reindex(right_dates) if not right_df.empty else pd.DataFrame(index=right_dates)
right_tbl["sales_million"] = (pd.to_numeric(r_vals.get("sales_amount"), errors="coerce")/1_000_000).values
right_tbl["passengers_k"]  = (pd.to_numeric(r_vals.get("passengers"),   errors="coerce")/1_000).values

merged_tbl = pd.merge(
    right_tbl,
    left_tbl[["pos","sales_million","passengers_k"]] if not left_tbl.empty else pd.DataFrame(columns=["pos"]),
    on="pos", how="left", suffixes=("_f","_a")
)

for c in ["sales_million_f","sales_million_a","passengers_k_f","passengers_k_a"]:
    merged_tbl[c] = pd.to_numeric(merged_tbl.get(c), errors="coerce")

old_sales = merged_tbl["sales_million_a"]; new_sales = merged_tbl["sales_million_f"]
merged_tbl["sales_pct"] = np.where((old_sales.isna()) | (old_sales==0), np.nan, (new_sales-old_sales)/old_sales*100.0)
old_pax = merged_tbl["passengers_k_a"]; new_pax = merged_tbl["passengers_k_f"]
merged_tbl["pax_pct"] = np.where((old_pax.isna()) | (old_pax==0), np.nan, (new_pax-old_pax)/old_pax*100.0)

# === 휴일 라벨/툴팁 (실적/예측 모두 생성)
left_holiday_labels, left_holiday_fulls = ([], [])
right_holiday_labels, right_holiday_fulls = ([], [])
if len(left_dates) > 0:
    left_holiday_labels, left_holiday_fulls = build_holiday_labels(left_dates, holidays_df, max_len=6)
right_holiday_labels, right_holiday_fulls = build_holiday_labels(right_dates, holidays_df, max_len=6)

# ---- 표 데이터 구성 ----
actual_df_show = pd.DataFrame()
if not left_tbl.empty:
    d = {
        "실적|일자": fmt_date_ko(left_tbl["date"]),
        "실적|매출액(백만원)": left_tbl["sales_million"].round(0).astype("Int64") if st.session_state.get("cb_sales", True) else pd.NA,
        "실적|승객수(천명)" : left_tbl["passengers_k"].round(0).astype("Int64") if st.session_state.get("cb_pax", True) else pd.NA,
        "실적|외부요인"     : left_event_strings,  # ← 클릭 대상 (이벤트 N)
        "실적|휴무여부"     : left_holiday_labels,
        "실적|휴일명(풀)"   : left_holiday_fulls,
    }
    actual_df_show = pd.DataFrame({k:v for k,v in d.items()
                                   if not (isinstance(v, pd.Series) and v.isna().all())})

fcast_dict = {
    "예측|일자": fmt_date_ko(right_tbl["date"]),
}
if st.session_state.get("cb_sales", True):
    fcast_dict["예측|매출액(백만원(Δ))"] = [
        f"{int(round(v if not pd.isna(v) else 0)):,.0f}" + (f" ({'▲' if (not pd.isna(p) and p>=0) else ('▼' if not pd.isna(p) else '')}{'' if pd.isna(p) else f'{abs(p):.1f}%'} )" if not pd.isna(p) else "")
        for v,p in zip(right_tbl["sales_million"], merged_tbl["sales_pct"])
    ]
if st.session_state.get("cb_pax", True):
    fcast_dict["예측|승객수(천명(Δ))"] = [
        f"{int(round(v if not pd.isna(v) else 0)):,.0f}" + (f" ({'▲' if (not pd.isna(p) and p>=0) else ('▼' if not pd.isna(p) else '')}{'' if pd.isna(p) else f'{abs(p):.1f}%'} )" if not pd.isna(p) else "")
        for v,p in zip(right_tbl["passengers_k"], merged_tbl["pax_pct"])
    ]
fcast_dict["예측|휴무여부"]   = right_holiday_labels
fcast_dict["예측|휴일명(풀)"] = right_holiday_fulls

forecast_df_show = pd.DataFrame(fcast_dict)

# 합치기
if not actual_df_show.empty:
    table_df = pd.concat([actual_df_show, forecast_df_show], axis=1)
else:
    table_df = forecast_df_show.copy()

# 합계행(첫 행)
left_sales_m   = int(round(left_tbl["sales_million"].sum())) if "sales_million" in left_tbl.columns else 0
left_pax_k     = int(round(left_tbl["passengers_k"].sum()))   if "passengers_k"  in left_tbl.columns else 0
right_sales_m2 = int(round(right_tbl["sales_million"].sum())) if "sales_million" in right_tbl.columns else 0
right_pax_k2   = int(round(right_tbl["passengers_k"].sum()))  if "passengers_k"  in right_tbl.columns else 0

def top_delta_str(val, base):
    if base is None or base == 0 or val is None: return ""
    delta = (val - base) / base * 100.0; arrow = "▲" if delta >= 0 else "▼"
    return f" ({arrow}{abs(delta):.1f}%)"

sum_row = {}
# 실적 합계
if "실적|일자" in table_df.columns: sum_row["실적|일자"] = "합계"
if "실적|매출액(백만원)" in table_df.columns: sum_row["실적|매출액(백만원)"] = left_sales_m
if "실적|승객수(천명)"  in table_df.columns: sum_row["실적|승객수(천명)"]  = left_pax_k
if "실적|외부요인"      in table_df.columns: sum_row["실적|외부요인"]      = ""
if "실적|휴무여부"      in table_df.columns: sum_row["실적|휴무여부"]      = ""
if "실적|휴일명(풀)"    in table_df.columns: sum_row["실적|휴일명(풀)"]    = ""
# 예측 합계
if "예측|일자" in table_df.columns: sum_row["예측|일자"] = "합계"
if "예측|매출액(백만원(Δ))" in table_df.columns:
    sum_row["예측|매출액(백만원(Δ))"] = f"{right_sales_m2:,.0f}{top_delta_str(right_sales_m2, left_sales_m if left_sales_m>0 else None)}"
if "예측|승객수(천명(Δ))" in table_df.columns:
    sum_row["예측|승객수(천명(Δ))"]   = f"{right_pax_k2:,.0f}{top_delta_str(right_pax_k2, left_pax_k if left_pax_k>0 else None)}"
if "예측|휴무여부" in table_df.columns: sum_row["예측|휴무여부"] = ""
if "예측|휴일명(풀)" in table_df.columns: sum_row["예측|휴일명(풀)"] = ""

# ======== 매트릭스 렌더링 (실적/예측 분리) ========
st.markdown("#### 📋 데이터 표")

# ---- 실적/예측 매트릭스 생성 함수 (동일)
def _build_left_matrix() -> pd.DataFrame:
    if left_tbl.empty:
        return pd.DataFrame()
    rows = {}
    if st.session_state.get("cb_sales", True) and "sales_million" in left_tbl:
        rows["매출액(백만원)|실적"] = left_tbl["sales_million"].round(0).astype("Int64").tolist()
    if st.session_state.get("cb_pax", True) and "passengers_k" in left_tbl:
        rows["승객수(천명)|실적"] = left_tbl["passengers_k"].round(0).astype("Int64").tolist()
    df = pd.DataFrame.from_dict(rows, orient="index", columns=fmt_date_ko(left_tbl["date"]))
    sums = []
    for idx in df.index:
        s = pd.to_numeric(df.loc[idx], errors="coerce").sum(min_count=1)
        sums.append("" if pd.isna(s) else int(round(s)))
    df.insert(0, "합계", sums)
    return df

def _build_right_matrix() -> pd.DataFrame:
    if right_tbl.empty:
        return pd.DataFrame()
    def _delta_str(pct):
        if pd.isna(pct): return ""
        return f" ({'▲' if pct>=0 else '▼'}{abs(pct):.1f}%)"
    rows = {}
    if st.session_state.get("cb_sales", True) and "sales_million" in right_tbl:
        vals = right_tbl["sales_million"]
        pcts = merged_tbl["sales_pct"] if "sales_pct" in merged_tbl.columns else pd.Series([pd.NA]*len(vals))
        rows["매출액(백만원)|예측(Δ)"] = [
            ("" if pd.isna(v) else f"{int(round(v)):,.0f}") + _delta_str(p)
            for v, p in zip(vals, pcts)
        ]
    if st.session_state.get("cb_pax", True) and "passengers_k" in right_tbl:
        vals = right_tbl["passengers_k"]
        pcts = merged_tbl["pax_pct"] if "pax_pct" in merged_tbl.columns else pd.Series([pd.NA]*len(vals))
        rows["승객수(천명)|예측(Δ)"] = [
            ("" if pd.isna(v) else f"{int(round(v)):,.0f}") + _delta_str(p)
            for v, p in zip(vals, pcts)
        ]
    df = pd.DataFrame.from_dict(rows, orient="index", columns=fmt_date_ko(right_tbl["date"]))
    sum_col = []
    for idx in df.index:
        if idx.startswith("매출액"):
            s = pd.to_numeric(right_tbl.get("sales_million"), errors="coerce").sum(min_count=1)
            sum_col.append("" if pd.isna(s) else f"{int(round(s)):,.0f}")
        elif idx.startswith("승객수"):
            s = pd.to_numeric(right_tbl.get("passengers_k"), errors="coerce").sum(min_count=1)
            sum_col.append("" if pd.isna(s) else f"{int(round(s)):,.0f}")
        else:
            sum_col.append("")
    df.insert(0, "합계", sum_col)
    return df

left_matrix  = _build_left_matrix()
right_matrix = _build_right_matrix()

def _transpose_with_sum_first(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    t = df.T
    if "합계" in t.index:
        t = pd.concat([t.loc[["합계"]], t.drop(index=["합계"])], axis=0)
    return t

left_T  = _transpose_with_sum_first(left_matrix)
right_T = _transpose_with_sum_first(right_matrix)

# ==== 전치표: 외부요인/휴일 컬럼 추가 및 포맷 ====
if not left_T.empty and not left_tbl.empty:
    ext_values = build_event_strings(pd.DatetimeIndex(left_tbl["date"]), external_factors_df)
    left_holiday_labels2, _ = build_holiday_labels(pd.DatetimeIndex(left_tbl["date"]), holidays_df, max_len=6)

    def _append_aligned_column(T: pd.DataFrame, dates: pd.Series, values: list, col_name: str):
        if T is None or T.empty: return T
        date_labels = list(fmt_date_ko(pd.Series(dates)))
        mapping = {lbl: val for lbl, val in zip(date_labels, values)}
        aligned = []
        for idx in T.index:
            aligned.append("" if str(idx) == "합계" else mapping.get(idx, ""))
        T[col_name] = aligned
        return T

    left_T = _append_aligned_column(left_T, left_tbl["date"], ext_values, "외부요인")
    left_T = _append_aligned_column(left_T, left_tbl["date"], left_holiday_labels2, "휴일")

    # 숫자 포맷(3자리 콤마)
    def _fmt_commas(v):
        if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == ""):
            return ""
        try:
            n = pd.to_numeric(v, errors="coerce")
            if pd.isna(n):
                return str(v)
            return f"{int(round(n)):,}"
        except Exception:
            return str(v)
    num_cols = [c for c in left_T.columns if ("매출액" in c) or ("승객수" in c)]
    for c in num_cols:
        left_T[c] = left_T[c].apply(_fmt_commas)

# ==== 예측기간(전치): 휴일 컬럼만 맨 끝에 ====
if not right_T.empty and not right_tbl.empty:
    def _append_aligned_column(T: pd.DataFrame, dates: pd.Series, values: list, col_name: str):
        if T is None or T.empty: return T
        date_labels = list(fmt_date_ko(pd.Series(dates)))
        mapping = {lbl: val for lbl, val in zip(date_labels, values)}
        aligned = []
        for idx in T.index:
            aligned.append("" if str(idx) == "합계" else mapping.get(idx, ""))
        T[col_name] = aligned
        return T
    if "휴무" in right_T.columns:
        right_T = right_T.drop(columns=["휴무"])
    right_holiday_labels2, _ = build_holiday_labels(pd.DatetimeIndex(right_tbl["date"]), holidays_df, max_len=6)
    if "휴일" in right_T.columns:
        _col = right_T.pop("휴일")
        right_T["휴일"] = _col
    else:
        right_T = _append_aligned_column(right_T, right_tbl["date"], right_holiday_labels2, "휴일")

# ==== (중요) 최초 진입/기간 변경 시 이벤트 맵 선생성 ====
@st.cache_data(show_spinner=False)
def load_concert_counts_df() -> pd.DataFrame:
    try:
        df = load_df_from_repo_csv("merged.csv").copy()
    except FileNotFoundError:
        st.warning("'merged.csv'를 찾지 못해 콘서트 카운트를 표시할 수 없습니다.")
        return pd.DataFrame(columns=["date","concerts_events_count"])
    cols = {c.lower(): c for c in df.columns}
    need = ["date","concerts_events_count"]
    for k in need:
        if k not in [c.lower() for c in df.columns]:
            st.warning(f"'merged.csv'에 '{k}' 컬럼이 없습니다.")
            return pd.DataFrame(columns=["date","concerts_events_count"])
    df.rename(columns={cols.get("date","date"):"date",
                       cols.get("concerts_events_count","concerts_events_count"):"concerts_events_count"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["concerts_events_count"] = pd.to_numeric(df["concerts_events_count"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[(df["date"] >= ACT_START) & (df["date"] <= ACT_END)]
    return df[["date","concerts_events_count"]]

@st.cache_data(show_spinner=False)
def load_concert_info_df() -> pd.DataFrame:
    try:
        df = load_df_from_repo_csv("concert_info_rows.csv").copy()
    except FileNotFoundError:
        st.warning("'concert_info_rows.csv' 파일을 찾을 수 없습니다. 콘서트 상세를 건너뜁니다.")
        return pd.DataFrame(columns=["title","start_date","end_date","label"])
    required = ["title","s_y","s_m","s_d","e_y","e_m","e_d"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"'concert_info_rows.csv'에 다음 컬럼이 필요합니다: {', '.join(missing)}")
        return pd.DataFrame(columns=["title","start_date","end_date","label"])

    def _to_int(s):
        return pd.to_numeric(pd.Series(s, dtype=str).str.replace(r"[^\d]", "", regex=True), errors="coerce").astype("Int64")

    for c in ["s_y","s_m","s_d","e_y","e_m","e_d"]:
        df[c] = _to_int(df[c])

    s_str = (df["s_y"].astype(str).str.zfill(4) + "-" +
             df["s_m"].astype(str).str.zfill(2) + "-" +
             df["s_d"].astype(str).str.zfill(2))
    e_str = (df["e_y"].astype(str).str.zfill(4) + "-" +
             df["e_m"].astype(str).str.zfill(2) + "-" +
             df["e_d"].astype(str).str.zfill(2))

    df["start_date"] = pd.to_datetime(s_str, errors="coerce")
    df["end_date"]   = pd.to_datetime(e_str, errors="coerce")
    df = df.dropna(subset=["start_date","end_date"])
    df = df[df["start_date"] <= df["end_date"]].copy()
    df["label"] = "(Concert)" + df["title"].astype(str) + " (" + df["start_date"].dt.strftime("%Y-%m-%d") + "~" + df["end_date"].dt.strftime("%Y-%m-%d") + ")"
    return df[["title","start_date","end_date","label"]]

def build_concert_map_by_date(visible_dates: pd.DatetimeIndex,
                              counts_df: pd.DataFrame,
                              info_df: pd.DataFrame) -> dict:
    if visible_dates is None or len(visible_dates) == 0 or counts_df is None or counts_df.empty or info_df is None or info_df.empty:
        return {}
    counts = counts_df.set_index("date").reindex(visible_dates).fillna(0)
    target_days = [d.normalize() for d in visible_dates if counts.loc[d, "concerts_events_count"] > 0]
    if not target_days:
        return {}
    target_set = set(target_days)
    bucket = {d: [] for d in target_days}
    min_d, max_d = min(target_set), max(target_set)
    for _, row in load_concert_info_df().iterrows() if info_df is None else info_df.iterrows():
        s, e, label = row["start_date"].normalize(), row["end_date"].normalize(), str(row["label"])
        if pd.isna(s) or pd.isna(e) or label.strip() == "": continue
        if e < min_d or s > max_d: continue
        start = max(s, min_d); end = min(e, max_d)
        for d in pd.date_range(start, end, freq="D"):
            d0 = d.normalize()
            if d0 in target_set:
                bucket[d0].append(label)
    bucket = {d: [t for t in titles if t.strip() != ""] for d, titles in bucket.items()}
    return {d: titles for d, titles in bucket.items() if titles}

@st.cache_data(show_spinner=False)
def load_expo_counts_df() -> pd.DataFrame:
    try:
        df = load_df_from_repo_csv("merged.csv").copy()
    except FileNotFoundError:
        st.warning("'merged.csv'를 찾지 못해 박람회 카운트를 표시할 수 없습니다.")
        return pd.DataFrame(columns=["date","coex_events_count","kintex_events_count","bexco_events_count"])
    cols = {c.lower(): c for c in df.columns}
    need = ["date","coex_events_count","kintex_events_count","bexco_events_count"]
    for k in need:
        if k not in [c.lower() for c in df.columns]:
            st.warning(f"'merged.csv'에 '{k}' 컬럼이 없습니다.")
            return pd.DataFrame(columns=["date","coex_events_count","kintex_events_count","bexco_events_count"])
    df.rename(columns={
        cols.get("date","date"): "date",
        cols.get("coex_events_count","coex_events_count"): "coex_events_count",
        cols.get("kintex_events_count","kintex_events_count"): "kintex_events_count",
        cols.get("bexco_events_count","bexco_events_count"): "bexco_events_count",
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["coex_events_count","kintex_events_count","bexco_events_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[(df["date"] >= ACT_START) & (df["date"] <= ACT_END)]
    return df[["date","coex_events_count","kintex_events_count","bexco_events_count"]]

@st.cache_data(show_spinner=False)
def load_expo_info_df(file_name: str, venue_prefix: str) -> pd.DataFrame:
    try:
        df = load_df_from_repo_csv(file_name).copy()
    except FileNotFoundError:
        st.warning(f"'{file_name}' 파일을 찾을 수 없습니다. ({venue_prefix} 상세 건너뜀)")
        return pd.DataFrame(columns=["event_name","start_date","end_date","label"])
    if "event_name" not in df.columns:
        st.warning(f"'{file_name}'에 'event_name' 컬럼이 없습니다.")
        df["event_name"] = ""
    start_col = "start_date" if "start_date" in df.columns else ("strart_date" if "strart_date" in df.columns else None)
    end_col   = "end_date" if "end_date" in df.columns else None
    if start_col is None or end_col is None:
        st.warning(f"'{file_name}'에 시작/종료일 컬럼(start_date/strart_date, end_date)이 없습니다.")
        return pd.DataFrame(columns=["event_name","start_date","end_date","label"])
    df["start_date"] = pd.to_datetime(df[start_col], errors="coerce")
    df["end_date"]   = pd.to_datetime(df[end_col],   errors="coerce")
    df = df.dropna(subset=["start_date","end_date"])
    df = df[df["start_date"] <= df["end_date"]].copy()
    df["event_name"] = df["event_name"].astype(str)
    df["label"] = "(" + venue_prefix + ")" + df["event_name"] + " (" + \
                  df["start_date"].dt.strftime("%Y-%m-%d") + "~" + df["end_date"].dt.strftime("%Y-%m-%d") + ")"
    return df[["event_name","start_date","end_date","label"]]

def build_event_titles_by_date(visible_dates: pd.DatetimeIndex,
                               counts_df: pd.DataFrame,
                               info_df: pd.DataFrame,
                               count_col: str) -> dict:
    if visible_dates is None or len(visible_dates) == 0:
        return {}
    if counts_df is None or counts_df.empty or info_df is None or info_df.empty:
        return {}
    counts = counts_df.set_index("date").reindex(visible_dates).fillna(0)
    target_days = [d.normalize() for d in visible_dates if (count_col in counts.columns and counts.loc[d, count_col] > 0)]
    if not target_days:
        return {}
    target_set = set(target_days)
    bucket = {d: [] for d in target_days}
    min_d, max_d = min(target_set), max(target_set)
    for _, row in info_df.iterrows():
        s, e, label = pd.to_datetime(row["start_date"]).normalize(), pd.to_datetime(row["end_date"]).normalize(), str(row["label"])
        if pd.isna(s) or pd.isna(e) or label.strip() == "":
            continue
        if e < min_d or s > max_d:
            continue
        start = max(s, min_d); end = min(e, max_d)
        if start > end:
            continue
        for d in pd.date_range(start, end, freq="D"):
            d0 = d.normalize()
            if d0 in target_set:
                bucket[d0].append(label)
    bucket = {d: [t for t in titles if t.strip() != ""] for d, titles in bucket.items()}
    bucket = {d: titles for d, titles in bucket.items() if titles}
    return bucket

@st.cache_data(show_spinner=False)
def load_sports_counts_df() -> pd.DataFrame:
    try:
        df = load_df_from_repo_csv("merged.csv").copy()
    except FileNotFoundError:
        st.warning("'merged.csv'를 찾지 못해 스포츠 카운트를 표시할 수 없습니다.")
        return pd.DataFrame(columns=["date","games_baseball","games_soccer"])
    cols = {c.lower(): c for c in df.columns}
    need = ["date","games_baseball","games_soccer"]
    for k in need:
        if k not in [c.lower() for c in df.columns]:
            st.warning(f"'merged.csv'에 '{k}' 컬럼이 없습니다.")
            return pd.DataFrame(columns=["date","games_baseball","games_soccer"])
    df.rename(columns={
        cols.get("date","date"): "date",
        cols.get("games_baseball","games_baseball"): "games_baseball",
        cols.get("games_soccer","games_soccer"): "games_soccer",
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["games_baseball","games_soccer"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[(df["date"] >= ACT_START) & (df["date"] <= ACT_END)]
    return df[["date","games_baseball","games_soccer"]]

@st.cache_data(show_spinner=False)
def load_baseball_schedule_df() -> pd.DataFrame:
    try:
        df = load_df_from_repo_csv("baseball_schedule_rows.csv").copy()
    except FileNotFoundError:
        st.warning("'baseball_schedule_rows.csv' 파일을 찾을 수 없습니다. (야구 일정 생략)")
        return pd.DataFrame(columns=["date","label"])
    required = ["game_date","home_team","away_team","region"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"야구 일정에 필요한 컬럼이 없습니다: {', '.join(missing)}")
        return pd.DataFrame(columns=["date","label"])
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["game_date"]).copy()
    if "note" in df.columns:
        mask_keep = df["note"].isna() | (df["note"].astype(str).str.strip() == "")
        df = df[mask_keep].copy()
    for c in ["home_team","away_team","region"]:
        df[c] = df[c].astype(str).fillna("")
    df["label"] = "(Baseball)" + df["home_team"] + " VS " + df["away_team"] + " in " + df["region"]
    df.rename(columns={"game_date":"date"}, inplace=True)
    return df[["date","label"]]

@st.cache_data(show_spinner=False)
def load_kleague_schedule_df() -> pd.DataFrame:
    try:
        df = load_df_from_repo_csv("k_league_rows.csv").copy()
    except FileNotFoundError:
        st.warning("'k_league_rows.csv' 파일을 찾을 수 없습니다. (K리그 일정 생략)")
        return pd.DataFrame(columns=["date","label"])
    for c in ["date","class","stadium"]:
        if c not in df.columns:
            df[c] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).copy()
    df["class"]   = df["class"].astype(str)
    df["stadium"] = df["stadium"].astype(str)
    df["label"] = "(K-league)" + df["class"] + " in " + df["stadium"]
    return df[["date","label"]]

def build_single_day_titles_by_date(visible_dates: pd.DatetimeIndex,
                                    counts_df: pd.DataFrame,
                                    info_df: pd.DataFrame,
                                    count_col: str,
                                    info_date_col: str = "date") -> dict:
    if visible_dates is None or len(visible_dates) == 0:
        return {}
    if counts_df is None or counts_df.empty or info_df is None or info_df.empty:
        return {}
    counts = counts_df.set_index("date").reindex(visible_dates).fillna(0)
    target_days = [d.normalize() for d in visible_dates if (count_col in counts.columns and counts.loc[d, count_col] > 0)]
    if not target_days:
        return {}
    info_df = info_df.copy()
    info_df[info_date_col] = pd.to_datetime(info_df[info_date_col]).dt.normalize()
    by_date = (info_df.groupby(info_date_col)["label"]
                     .apply(lambda s: [str(x) for x in s if isinstance(x, str) and x.strip() != ""])
                     .to_dict())
    bucket = {}
    for d in target_days:
        labels = by_date.get(d, [])
        if labels:
            bucket[d] = labels
    return bucket

def _build_integrated_map_for_range(s: pd.Timestamp, e: pd.Timestamp) -> dict:
    if s is None or e is None or s > e:
        return {}
    visible_left = pd.date_range(s, e, freq="D")
    concert_counts_df = load_concert_counts_df()
    concert_info_df   = load_concert_info_df()
    concert_map = build_concert_map_by_date(visible_left, concert_counts_df, concert_info_df)
    expo_counts_df = load_expo_counts_df()
    coex_info_df   = load_expo_info_df("coex_events_rows.csv",   "Coex")
    kintex_info_df = load_expo_info_df("kintex_events_rows.csv", "Kintex")
    bexco_info_df  = load_expo_info_df("bexco_events_rows.csv",  "Bexco")
    coex_map   = build_event_titles_by_date(visible_left, expo_counts_df, coex_info_df,   "coex_events_count")
    kintex_map = build_event_titles_by_date(visible_left, expo_counts_df, kintex_info_df, "kintex_events_count")
    bexco_map  = build_event_titles_by_date(visible_left, expo_counts_df, bexco_info_df,  "bexco_events_count")
    sports_counts_df = load_sports_counts_df()
    baseball_df = load_baseball_schedule_df()
    kleague_df  = load_kleague_schedule_df()
    baseball_map = build_single_day_titles_by_date(visible_left, sports_counts_df, baseball_df, "games_baseball", info_date_col="date")
    kleague_map  = build_single_day_titles_by_date(visible_left, sports_counts_df, kleague_df,  "games_soccer",   info_date_col="date")
    out = {}
    for d in visible_left:
        d0 = d.normalize()
        items = []
        items += concert_map.get(d0, [])
        items += coex_map.get(d0, [])
        items += kintex_map.get(d0, [])
        items += bexco_map.get(d0, [])
        items += baseball_map.get(d0, [])
        items += kleague_map.get(d0, [])
        if items:
            out[d0] = items
    return out

# ── 세션에 없거나 기간 바뀌면 선생성
_prev = st.session_state.get("_evt_map_range", None)
_cur = (l_s, l_e)
need_build = ("integrated_event_map" not in st.session_state) or (_prev != _cur)
if need_build:
    st.session_state["integrated_event_map"] = _build_integrated_map_for_range(l_s, l_e)
    st.session_state["_evt_map_range"] = _cur

# ---- (도우미) 전치 테이블의 인덱스에 맞춰 안전하게 컬럼 추가 (재사용용)
def _append_aligned_column(T: pd.DataFrame, dates: pd.Series, values: list, col_name: str):
    if T is None or T.empty:
        return T
    date_labels = list(fmt_date_ko(pd.Series(dates)))
    mapping = {lbl: val for lbl, val in zip(date_labels, values)}
    aligned = []
    for idx in T.index:
        if str(idx) == "합계":
            aligned.append("")
        else:
            aligned.append(mapping.get(idx, ""))
    T[col_name] = aligned
    return T

# ---- 스타일 (주말 색) ── st.data_editor로 변경하면서 사용 X, 필요 시 컬럼으로 대체 가능
def _style_weekend_rows(df: pd.DataFrame) -> Styler:
    blue_text = "#1e90ff"; red_text = "#ef4444"
    sty = df.style.set_properties(**{"text-align":"center"}).set_table_styles([{"selector":"th","props":"text-align:center;"}])
    if "합계" in df.index:
        sty = sty.set_properties(subset=(["합계"], df.columns), **{"font-weight":"bold","background-color": SUM_BG})
    for idx in df.index:
        if isinstance(idx, str) and "(토)" in idx:
            sty = sty.set_properties(subset=([idx], df.columns), **{"color": blue_text})
        if isinstance(idx, str) and "(일)" in idx:
            sty = sty.set_properties(subset=([idx], df.columns), **{"color": red_text})
    return sty

# ---- 출력 (전치표 + 체크박스 + 선택된 일자 이벤트) ----
c1, c2 = st.columns(2)
with c1:
    st.markdown("**✅실적**")
    if left_T.empty:
        st.info("실적 기간 데이터가 없습니다.")
    else:
        # 인덱스('일자')를 컬럼으로 꺼내고 '외부요인' 옆에 체크박스 추가
        left_T.index.name = "일자"
        left_edit = left_T.reset_index()

        insert_pos = left_edit.columns.get_loc("외부요인") + 1 if "외부요인" in left_edit.columns else len(left_edit.columns)
        if "선택" not in left_edit.columns:
            left_edit.insert(insert_pos, "선택", False)

        edited_left = st.data_editor(
            left_edit,
            hide_index=True,
            use_container_width=True,
            height=min(520, 140 + 28 * max(3, len(left_edit))),
            column_config={
                "선택": st.column_config.CheckboxColumn(
                    "선택", help="해당 일자의 이벤트를 선택합니다.", default=False,
                ),
            },
            disabled=["일자"],  # 날짜 수정 방지
        )

        # ✅ 체크된 날짜 저장 (합계 제외) — 상세보기 표는 아래 전용 섹션에서 전체폭으로 렌더링
        selected_mask = (edited_left.get("선택") == True) & (edited_left.get("일자") != "합계")
        st.session_state["selected_event_dates_from_matrix"] = edited_left.loc[selected_mask, "일자"].tolist()

with c2:
    st.markdown("**✅예측**")
    if right_T.empty:
        st.info("예측 기간 데이터가 없습니다.")
    else:
        right_T.index.name = "일자"
        st.dataframe(_style_weekend_rows(right_T), use_container_width=True,
                     height=min(520, 140 + 28 * max(3, len(right_T))))

# ===================== 🔎 외부요인 상세보기 (전체 폭) =====================
st.markdown("#### 🔎 외부요인 상세보기")

def _label_to_date(lbl: str):
    try:
        s = str(lbl).strip()
        iso = s[:10]
        dt = pd.to_datetime(iso, errors="coerce")
        return None if pd.isna(dt) else dt.normalize()
    except Exception:
        return None

_selected_labels = st.session_state.get("selected_event_dates_from_matrix", []) or []
_selected_dates = [d for d in (_label_to_date(x) for x in _selected_labels) if d is not None]
_selected_dates = sorted(set(_selected_dates))

integrated_map = st.session_state.get("integrated_event_map", {})

if not _selected_dates:
    st.info("체크한 날짜가 없습니다.")
else:
    def build_event_detail_df(selected_dates: list[pd.Timestamp], event_map: dict) -> dict:
        """선택된 날짜별 개별 DataFrame 생성"""
        result = {}
        for d0 in selected_dates:
            pretty = fmt_date_ko(pd.Series([d0])).iloc[0]
            events = event_map.get(d0, [])
            rows = []

            if not events:
                rows.append({"카테고리": "", "이벤트": "(표시할 이벤트가 없습니다)", "기간": "", "비고": ""})
            else:
                for t in events:
                    raw = str(t).strip()
                    # (카테고리) 제목 (2024-01-01~2024-01-03) 형태 분리
                    m = re.match(r"^\(([^)]+)\)\s*(.*)$", raw)
                    cat = m.group(1) if m else ""
                    title = m.group(2) if m else raw

                    # 카테고리 변환
                    if cat.lower() == "concert":
                        cat_kr = "콘서트"
                    elif cat.lower() in ["coex", "bexco", "kintex"]:
                        cat_kr = "박람회"
                    elif cat.lower() in ["baseball", "k-league"]:
                        cat_kr = "스포츠"
                    else:
                        cat_kr = cat

                    # 기간 추출
                    period_match = re.search(r"\((\d{4}-\d{2}-\d{2}~\d{4}-\d{2}-\d{2})\)", title)
                    period = period_match.group(1) if period_match else ""
                    title_clean = re.sub(r"\(\d{4}-\d{2}-\d{2}~\d{4}-\d{2}-\d{2}\)", "", title).strip()

                    # 콘서트 외 카테고리는 "(카테고리)" 접두어 추가
                    if cat_kr != "콘서트" and cat:
                        title_clean = f"({cat}) " + title_clean

                    rows.append({
                        "카테고리": cat_kr,
                        "이벤트": title_clean,
                        "기간": period,
                        "비고": ""
                    })
            result[d0] = pd.DataFrame(rows, columns=["카테고리", "이벤트", "기간", "비고"])
        return result

    # 날짜별 개별 표 생성
    detail_map = build_event_detail_df(_selected_dates, integrated_map)

    # 각 날짜별 표를 개별로 렌더링
    for d0 in _selected_dates:
        df_day = detail_map.get(d0, pd.DataFrame())
        pretty = fmt_date_ko(pd.Series([d0])).iloc[0]
        st.markdown(f"**📅 {pretty}**")
        if df_day.empty:
            st.info("이 날짜에는 표시할 이벤트가 없습니다.")
        else:
            st.dataframe(
                df_day,
                use_container_width=True,
                height=min(380, 120 + 28 * (len(df_day) + 1))
            )
        st.markdown("---")




# ===================== 9월 예측 정확도 (실적 vs 예측) =====================
st.markdown("#### 🎯 예측 정확도 (실적 vs 예측)")

SEP_START = pd.to_datetime("2025-09-01")
SEP_END   = pd.to_datetime("2025-09-30")

@st.cache_data(show_spinner=False)
def load_actual_sep_df() -> pd.DataFrame:
    SEP_START = pd.to_datetime("2025-09-01")
    SEP_END   = pd.to_datetime("2025-09-30")
    try:
        raw = load_df_from_repo_csv("actual_sep_rows.csv").copy()
    except FileNotFoundError:
        st.warning("'actual_sep_rows.csv' 파일을 찾을 수 없습니다. (9월 정확도 표 생략)")
        return pd.DataFrame(columns=["date","passengers","sales_amount"])
    if raw.empty:
        st.warning("actual_sep_rows.csv가 비어 있습니다.")
        return pd.DataFrame(columns=["date","passengers","sales_amount"])
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col_candidates = ["travel_date","date","일자","날짜","ts","dt"]
    ymd_candidates = [("year","month","day"),("yyyy","mm","dd"),("y","m","d"),("s_y","s_m","s_d")]
    date_series = None
    for c in date_col_candidates:
        if c in df.columns:
            date_series = pd.to_datetime(df[c], errors="coerce")
            break
    if date_series is None:
        for y,m,d in ymd_candidates:
            if y in df.columns and m in df.columns and d in df.columns:
                yv = pd.to_numeric(df[y], errors="coerce")
                mv = pd.to_numeric(df[m], errors="coerce")
                dv = pd.to_numeric(df[d], errors="coerce")
                date_series = pd.to_datetime(
                    yv.astype("Int64").astype(str).str.zfill(4) + "-" +
                    mv.astype("Int64").astype(str).str.zfill(2) + "-" +
                    dv.astype("Int64").astype(str).str.zfill(2),
                    errors="coerce"
                )
                break
    if date_series is None:
        st.warning("actual_sep_rows.csv에서 날짜 컬럼을 찾을 수 없습니다. (travel_date/date/일자/날짜 또는 연·월·일 조합 필요)")
        return pd.DataFrame(columns=["date","passengers","sales_amount"])
    df["date"] = pd.to_datetime(date_series, errors="coerce").dt.normalize()

    def _pick_col(cands):
        for c in cands:
            if c in df.columns: return c
        for c in df.columns:
            for key in cands:
                if key in c: return c
        return None
    pax_col = _pick_col(["passengers","pax","ridership","승객","승객수"])
    sales_col = _pick_col(["sales_amount","sales","revenue","amount","매출","매출액"])
    def to_numeric_clean(s):
        if s is None: return pd.Series(dtype="float64")
        return (pd.Series(s, dtype="object").astype(str)
                .str.replace(r"[,\s₩원$₩]", "", regex=True)
                .replace({"": np.nan, "nan": np.nan})
                .pipe(pd.to_numeric, errors="coerce"))
    df["passengers"] = to_numeric_clean(df[pax_col]) if pax_col else np.nan
    df["sales_amount"] = to_numeric_clean(df[sales_col]) if sales_col else np.nan
    df = df.dropna(subset=["date"])
    if df[["passengers","sales_amount"]].isna().all(axis=None):
        st.warning("actual_sep_rows.csv의 승객/매출 값이 모두 결측입니다.")
        return pd.DataFrame(columns=["date","passengers","sales_amount"])
    daily = (df.groupby("date", as_index=False)[["passengers","sales_amount"]]
               .sum(min_count=1).sort_values("date"))
    daily = daily[(daily["date"] >= SEP_START) & (daily["date"] <= SEP_END)]
    if daily.empty:
        st.info("actual_sep_rows.csv에서 2025-09 기간에 해당하는 데이터가 없습니다.")
        return pd.DataFrame(columns=["date","passengers","sales_amount"])
    return daily[["date","passengers","sales_amount"]]

def _safe_pct_err(forecast, actual):
    if pd.isna(actual) or actual == 0:
        return np.nan
    return (forecast - actual) / actual * 100.0

actual_sep = load_actual_sep_df()
fcst_pass_sep = forecast_df_all[(forecast_df_all["date"] >= SEP_START) & (forecast_df_all["date"] <= SEP_END)][["date","passengers"]].rename(columns={"passengers":"f_passengers"})
fcst_sales_sep = forecast_sales_all[(forecast_sales_all["date"] >= SEP_START) & (forecast_sales_all["date"] <= SEP_END)][["date","pred_sales_amount"]].rename(columns={"pred_sales_amount":"f_sales_amount"})
fcst_sep = pd.merge(fcst_pass_sep, fcst_sales_sep, on="date", how="outer").sort_values("date")
sel_start = max(r_s, SEP_START); sel_end = min(r_e, SEP_END)
cmp = pd.merge(fcst_sep, actual_sep, on="date", how="outer")
cmp = cmp[(cmp["date"] >= sel_start) & (cmp["date"] <= sel_end)].sort_values("date").reset_index(drop=True)
cmp["a_passengers"]   = pd.to_numeric(cmp.get("passengers"), errors="coerce")
cmp["a_sales_amount"] = pd.to_numeric(cmp.get("sales_amount"), errors="coerce")
cmp["f_passengers"]   = pd.to_numeric(cmp.get("f_passengers"), errors="coerce")
cmp["f_sales_amount"] = pd.to_numeric(cmp.get("f_sales_amount"), errors="coerce")
cmp["pax_err_pct"]   = [ _safe_pct_err(fp, ap) for fp, ap in zip(cmp["f_passengers"], cmp["a_passengers"]) ]
cmp["sales_err_pct"] = [ _safe_pct_err(fs, as_) for fs, as_ in zip(cmp["f_sales_amount"], cmp["a_sales_amount"]) ]

disp = pd.DataFrame({
    "일자": fmt_date_ko(cmp["date"].dt.tz_localize(None)) if "date" in cmp.columns else pd.Series(dtype=str),
    "실적|매출액(백만원)":  (cmp["a_sales_amount"] / 1_000_000).round(0).astype("Int64"),
    "예측|매출액(백만원)":  (cmp["f_sales_amount"] / 1_000_000).round(0).astype("Int64"),
    "오차율|매출액(%)":   cmp["sales_err_pct"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else ""),
    "실적|승객수(천명)":    (cmp["a_passengers"]  / 1_000).round(0).astype("Int64"),
    "예측|승객수(천명)":    (cmp["f_passengers"]  / 1_000).round(0).astype("Int64"),
    "오차율|승객수(%)":   cmp["pax_err_pct"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else ""),
})

def _mape_pct(f, a):
    s = [abs((fv - av)/av)*100.0 for fv, av in zip(f, a) if (not pd.isna(av) and av!=0 and not pd.isna(fv))]
    return np.nan if len(s)==0 else float(np.mean(s))

sum_a_pax   = cmp["a_passengers"].sum(skipna=True)
sum_f_pax   = cmp["f_passengers"].sum(skipna=True)
sum_a_sales = cmp["a_sales_amount"].sum(skipna=True)
sum_f_sales = cmp["f_sales_amount"].sum(skipna=True)
mape_pax   = _mape_pct(cmp["f_passengers"], cmp["a_passengers"])
mape_sales = _mape_pct(cmp["f_sales_amount"], cmp["a_sales_amount"])

sum_row2 = pd.DataFrame([{
    "일자": "합계",
    "실적|매출액(백만원)": int(round(sum_a_sales/1_000_000)) if not pd.isna(sum_a_sales) else pd.NA,
    "예측|매출액(백만원)": int(round(sum_f_sales/1_000_000)) if not pd.isna(sum_f_sales) else pd.NA,
    "오차율|매출액(%)":    round(mape_sales, 1) if not pd.isna(mape_sales) else pd.NA,
    "실적|승객수(천명)":   int(round(sum_a_pax/1_000)) if not pd.isna(sum_a_pax) else pd.NA,
    "예측|승객수(천명)":   int(round(sum_f_pax/1_000)) if not pd.isna(sum_f_pax) else pd.NA,
    "오차율|승객수(%)":    round(mape_pax, 1) if not pd.isna(mape_pax) else pd.NA,
}])

disp_out = pd.concat([sum_row2, disp], ignore_index=True)

def _weekday_textcolor_only_df(_df: pd.DataFrame) -> pd.DataFrame:
    blue_text = "#1e90ff"; red_text  = "#ef4444"
    styles = pd.DataFrame("", index=_df.index, columns=_df.columns)
    for i in _df.index[1:]:
        d = str(_df.at[i, "일자"]) if "일자" in _df.columns else ""
        if "(토)" in d:
            styles.loc[i, :] = [f"color:{blue_text};"] * styles.shape[1]
        if "(일)" in d:
            styles.loc[i, :] = [f"color:{red_text};"] * styles.shape[1]
    if 0 in styles.index:
        styles.loc[0, :] = [f"font-weight:bold; background-color:{SUM_BG};"] * styles.shape[1]
    return styles

# ==== 🎯 예측 정확도 표 수정 ====

# ==== 🎯 예측 정확도 표 수정 ====

# 숫자 포맷(천단위 콤마)
def fmt_thousand(v):
    try:
        if pd.isna(v): return ""
        return f"{int(round(v)):,}"
    except Exception:
        return str(v)

disp_out_fmt = disp_out.copy()

# 컬럼명 변경: '매출액', '승객수' 문구 제거
disp_out_fmt = disp_out_fmt.rename(columns={
    "실적|매출액(백만원)": "실적(백만원)",
    "예측|매출액(백만원)": "예측(백만원)",
    "오차율|매출액(%)": "오차율(%)",
    "실적|승객수(천명)": "실적(천명)",
    "예측|승객수(천명)": "예측(천명)",
    "오차율|승객수(%)": "오차율(%)_승객"
})

# 숫자열 천단위 콤마 적용
num_cols = ["실적(백만원)", "예측(백만원)", "실적(천명)", "예측(천명)"]
for c in num_cols:
    if c in disp_out_fmt.columns:
        disp_out_fmt[c] = disp_out_fmt[c].apply(fmt_thousand)

# 합계행 오차율 소수점 한자리 유지
for c in ["오차율(%)", "오차율(%)_승객"]:
    if c in disp_out_fmt.columns:
        val = str(disp_out_fmt.loc[0, c])
        if val.replace('.', '', 1).isdigit():
            disp_out_fmt.loc[0, c] = f"{float(val):.1f}"

# 왼쪽(매출), 오른쪽(승객) 표 분리
disp_sales = disp_out_fmt[["일자", "실적(백만원)", "예측(백만원)", "오차율(%)"]].copy()
disp_pax   = disp_out_fmt[["일자", "실적(천명)", "예측(천명)", "오차율(%)_승객"]].copy()
disp_pax   = disp_pax.rename(columns={"오차율(%)_승객": "오차율(%)"})

# 스타일 적용 (주말 색상, 합계 강조)
def _weekday_textcolor_only_df(_df: pd.DataFrame) -> pd.DataFrame:
    blue_text = "#1e90ff"; red_text  = "#ef4444"
    styles = pd.DataFrame("", index=_df.index, columns=_df.columns)
    for i in _df.index[1:]:
        d = str(_df.at[i, "일자"]) if "일자" in _df.columns else ""
        if "(토)" in d:
            styles.loc[i, :] = [f"color:{blue_text};"] * styles.shape[1]
        if "(일)" in d:
            styles.loc[i, :] = [f"color:{red_text};"] * styles.shape[1]
    if 0 in styles.index:
        styles.loc[0, :] = [f"font-weight:bold; background-color:{SUM_BG};"] * styles.shape[1]
    return styles

# 두 표 나란히 출력
col1, col2 = st.columns(2)
with col1:
    st.markdown("**💰 매출액**")
    st.dataframe(
        disp_sales.style
            .set_properties(**{"text-align":"center"})
            .set_table_styles([{"selector":"th","props":"text-align:center;"}])
            .apply(_weekday_textcolor_only_df, axis=None),
        use_container_width=True,
        height=min(520, 120 + 28 * (len(disp_sales)+1))
    )
with col2:
    st.markdown("**🚆 승객수**")
    st.dataframe(
        disp_pax.style
            .set_properties(**{"text-align":"center"})
            .set_table_styles([{"selector":"th","props":"text-align:center;"}])
            .apply(_weekday_textcolor_only_df, axis=None),
        use_container_width=True,
        height=min(520, 120 + 28 * (len(disp_pax)+1))
    )

