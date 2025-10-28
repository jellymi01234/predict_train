# app.py ── Streamlit (https://<YOUR-APP>.streamlit.app)

import io
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ================= 기본 설정 =================
st.set_page_config(page_title="Passengers & Sales (Dual Axis)", layout="wide")

# ======= 상단 타이틀 + 다크모드 토글 =======
title_col, theme_col = st.columns([1,0.18])
with title_col:
    st.title("📈 외부요인 기반 철도수요예측 시스템")
with theme_col:
    DARK = st.checkbox("🌙 다크 모드", value=False)

# ---------- 테마 색상 변수 ----------
if DARK:
    BG        = "#0B1220"
    SURFACE   = "#111827"
    PANEL_BG  = "#0F172A"
    BORDER    = "#1F2937"
    TEXT      = "#E5E7EB"
    SUBTEXT   = "#9CA3AF"
    SHADOW    = "rgba(0,0,0,0.35)"
    HILITE1   = "rgba(56,189,248,0.12)"  # 예측 영역 음영
    HILITE2   = "rgba(148,163,184,0.10)" # 실적 영역 음영
    CARD_BG   = "#111827"
    CARD_BORDER = "#374151"
    PLOTLY_TEMPLATE = "plotly_dark"
else:
    BG        = "#FFFFFF"
    SURFACE   = "#FFFFFF"
    PANEL_BG  = "#F8FAFC"
    BORDER    = "#E5E7EB"
    TEXT      = "#111827"
    SUBTEXT   = "#6B7280"
    SHADOW    = "rgba(0,0,0,0.05)"
    HILITE1   = "rgba(30,144,255,0.08)"
    HILITE2   = "rgba(100,116,139,0.06)"
    CARD_BG   = "#FFFFFF"
    CARD_BORDER = "#E5E7EB"
    PLOTLY_TEMPLATE = "plotly_white"

# ---- 글로벌 스타일: 간격/카드/패널/범례 ----
st.markdown(
    f"""
    <style>
    html, body, .stApp {{ background: {BG}; color:{TEXT}; }}
    .gap-xl {{ height: 16px; }}

    .panel {{
        border: 1px solid {BORDER};
        background: {PANEL_BG};
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 4px 18px {SHADOW};
        margin-bottom: 12px;
    }}
    .summary-card {{
        border: 1px solid {CARD_BORDER};
        background: {CARD_BG};
        border-radius: 10px;
        padding: 10px 12px;
        box-shadow: 0 1px 2px {SHADOW};
    }}
    .summary-title {{ font-size: 13px; color:{SUBTEXT}; margin:0 0 6px 0; }}
    .summary-value {{ font-weight: 800; font-size: 18px; margin:0; color:{TEXT}; }}
    .delta-up   {{ color: #60A5FA; font-weight: 800; }}
    .delta-down {{ color: #F87171; font-weight: 800; }}

    /* 범례 스와치 */
    .lg-line   {{ height:2px; border-top: 4px solid #1f77b4; border-radius:2px; display:inline-block; width:22px; margin-right:6px; }}
    .lg-line-dash {{
        height:0; display:inline-block; width:22px; margin-right:6px; position:relative; top:2px;
        border-top: 0; 
        background: linear-gradient(90deg, #1f77b4 45%, rgba(0,0,0,0) 45%) repeat-x;
        background-size: 8px 4px; background-position: 0 50%;
    }}
    .lg-bar    {{ background:#ff7f0e; display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; }}
    .lg-bar-f  {{ background:#ff7f0e; opacity:0.7; display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; }}
    .lg-text   {{ font-size: 13px; color:{TEXT}; vertical-align:middle; }}
    .legend-row {{ display:flex; gap:18px; align-items:center; flex-wrap:wrap; justify-content: flex-end; }}

    /* 데이터프레임 배경 대비 */
    div[data-testid="stDataFrame"] div[role="grid"] {{
        background: {SURFACE};
    }}
    </style>
    <div class="gap-xl"></div>
    """,
    unsafe_allow_html=True,
)

# ===== 기간 정의 =====
ACT_START = pd.to_datetime("2020-08-01")
ACT_END   = pd.to_datetime("2025-08-31")  # ✅ 실적은 여기까지만 사용
FCT_START = pd.to_datetime("2025-09-01")
FCT_END   = pd.to_datetime("2025-11-29")

# ================= 파일 로더 =================
@st.cache_data(show_spinner=False)
def load_df_from_repo_csv(filename: str):
    """CSV 파일을 utf-8-sig → cp949 순으로 읽기"""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"'{filename}' 파일을 찾을 수 없습니다.")
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

# ================= 실적 데이터 (merged.csv) =================
@st.cache_data(show_spinner=False)
def load_actual_df() -> pd.DataFrame:
    df = load_df_from_repo_csv("merged.csv").copy()
    cols = {c.lower(): c for c in df.columns}
    d = cols.get("date", "date")
    p = cols.get("passengers", "passengers")
    s = cols.get("sales_amount", "sales_amount")

    df.rename(columns={d: "date", p: "passengers", s: "sales_amount"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[(df["date"] >= ACT_START) & (df["date"] <= ACT_END)]  # ✅ 실적 범위 강제
    df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
    df["sales_amount"] = pd.to_numeric(df["sales_amount"], errors="coerce")
    return df

# ================= 예측 데이터 (forecast_pass.csv) =================
@st.cache_data(show_spinner=False)
def load_forecast_df() -> pd.DataFrame:
    f = load_df_from_repo_csv("forecast_pass.csv").copy()
    cols = {c.lower(): c for c in f.columns}
    d = cols.get("date", "date")
    p = cols.get("forecast_90d", "forecast_90d")

    f.rename(columns={d: "date", p: "passengers"}, inplace=True)
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f.dropna(subset=["date"]).sort_values("date")
    f = f[(f["date"] >= FCT_START) & (f["date"] <= FCT_END)]
    f["passengers"] = pd.to_numeric(f["passengers"], errors="coerce")
    f["sales_amount"] = np.nan
    return f

# ================= 예측 매출 로더 =================
@st.cache_data(show_spinner=False)
def load_forecast_sales_df() -> pd.DataFrame:
    try:
        f = load_df_from_repo_csv("forecast_sales.csv").copy()
    except FileNotFoundError:
        f = load_df_from_repo_csv("forecast_sales.cvs").copy()

    cols = {c.lower(): c for c in f.columns}
    d = cols.get("date", "date")
    v = cols.get("forecast_90d", "forecast_90d")

    f.rename(columns={d: "date", v: "pred_sales_amount"}, inplace=True)
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f.dropna(subset=["date"]).sort_values("date")
    f = f[(f["date"] >= FCT_START) & (f["date"] <= FCT_END)]
    f["pred_sales_amount"] = pd.to_numeric(f["pred_sales_amount"], errors="coerce")
    return f[["date", "pred_sales_amount"]]

# ================= 실적 rows 로더 =================
@st.cache_data(show_spinner=False)
def load_actual_rows_df() -> pd.DataFrame:
    df = load_df_from_repo_csv("train_reservations_rows.csv").copy()
    required = ["travel_date", "passengers", "sales_amount"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"'train_reservations_rows.csv'에 다음 컬럼이 필요합니다: {', '.join(miss)}")

    df["travel_date"] = pd.to_datetime(df["travel_date"], errors="coerce")
    df = df.dropna(subset=["travel_date"])
    for c in ["passengers", "sales_amount"]:
        df[c] = df[c].astype(str).str.replace(",", "", regex=False).replace("nan", np.nan)
        df[c] = pd.to_numeric(df[c], errors="coerce")

    daily = (
        df.assign(date=df["travel_date"].dt.floor("D"))
          .groupby("date", as_index=False)[["passengers", "sales_amount"]]
          .sum()
          .sort_values("date")
    )
    daily = daily[(daily["date"] >= ACT_START) & (daily["date"] <= ACT_END)]
    return daily

# ================= 유틸: 기간 보정/동일 길이/요일정렬 =================
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

# ===================== 사이드바: 기간 선택 =====================
st.sidebar.markdown("---")
st.sidebar.subheader("📅 기간 선택")

# ① 예측 기간
default_right_start = date(2025, 9, 1)
default_right_end   = date(2025, 9, 7)
right_range = st.session_state.get("right_range", (default_right_start, default_right_end))
right_sel = st.sidebar.date_input(
    "① 예측 기간 (YYYY-MM-DD)",
    value=right_range,
    min_value=FCT_START.date(), max_value=FCT_END.date(),
    key="right_picker_sidebar"
)

# ② 실적 기간 모드
left_mode = st.sidebar.radio(
    "② 실적 기간 모드",
    options=["사용 안 함 (예측만)", "전년도 동일(일자)", "전년도 동일(요일)", "사용자 지정"],
    index=1,
    key="left_mode_sidebar"
)

left_sel = None
if left_mode == "사용자 지정":
    left_range = st.session_state.get("left_range", (date(2024, 9, 1), date(2024, 9, 7)))
    left_sel = st.sidebar.date_input(
        "실적 기간 (YYYY-MM-DD)",
        value=left_range,
        min_value=ACT_START.date(), max_value=ACT_END.date(),
        key="left_picker_sidebar"
    )

# ================= 기간 정규화/동기화 =================
def norm_tuple(sel):
    if isinstance(sel, tuple):
        return sel[0], sel[1]
    return sel, sel

r_s, r_e = norm_tuple(right_sel)
r_s, r_e = pd.to_datetime(r_s), pd.to_datetime(r_e)
r_s, r_e = ensure_in_range(r_s, r_e, FCT_START, FCT_END)
N_days = (r_e - r_s).days + 1

if left_mode == "사용 안 함 (예측만)":
    l_s, l_e = None, None
elif left_mode == "전년도 동일(일자)":
    l_s = (r_s - pd.DateOffset(years=1)).normalize()
    l_e = l_s + pd.Timedelta(days=N_days-1)
    l_s, l_e = ensure_in_range(l_s, l_e, ACT_START, ACT_END)
elif left_mode == "전년도 동일(요일)":
    l_s, l_e = align_last_year_same_weekday(r_s, N_days)
else:  # 사용자 지정
    l_s, l_e = norm_tuple(left_sel)
    l_s, l_e = pd.to_datetime(l_s), pd.to_datetime(l_e)
    l_s, l_e, r_s, r_e = force_same_length(l_s, l_e, r_s, r_e)

# 세션 저장
st.session_state["right_range"] = (r_s.date(), r_e.date())
if left_mode == "사용자 지정" and l_s is not None:
    st.session_state["left_range"] = (l_s.date(), l_e.date())

# ================= 데이터 로드 & 가공 =================
actual_df_all   = load_actual_df()
forecast_df_all = load_forecast_df()
try:
    forecast_sales_all = load_forecast_sales_df()
except FileNotFoundError as e:
    st.warning(f"예측 매출 파일을 찾을 수 없어 매출 예측선을 그리지 못할 수 있습니다: {e}")
    forecast_sales_all = pd.DataFrame(columns=["date", "pred_sales_amount"])

def get_range(df, s, e, tag):
    if s is None or e is None:
        return pd.DataFrame(columns=["date","passengers","sales_amount","source"])
    out = df[(df["date"] >= s) & (df["date"] <= e)].copy()
    out["source"] = tag
    return out

left_df  = get_range(actual_df_all,   l_s, l_e, "actual") if l_s is not None else pd.DataFrame(columns=["date","passengers","sales_amount","source"])
right_df = get_range(forecast_df_all, r_s, r_e, "forecast")

# 예측 매출 주입
if not right_df.empty:
    right_df = right_df.merge(forecast_sales_all, on="date", how="left")
    right_df["sales_amount"] = np.where(
        right_df["sales_amount"].isna(), right_df["pred_sales_amount"], right_df["sales_amount"]
    )

# 병합 + 단위
df_sel = pd.concat(
    ([left_df.assign(period="실적기간")] if not left_df.empty else []) +
    [right_df.assign(period="예측기간")],
    ignore_index=True
).sort_values("date") if (not right_df.empty or not left_df.empty) else pd.DataFrame(columns=["date","passengers","sales_amount","source","period"])

df_sel["sales_million"] = pd.to_numeric(df_sel["sales_amount"], errors="coerce") / 1_000_000
df_sel["passengers_k"]  = pd.to_numeric(df_sel["passengers"], errors="coerce") / 1_000  # 천명

# ================= X축(두 블록 카테고리) =================
order_left  = pd.date_range(l_s, l_e, freq="D") if l_s is not None else pd.DatetimeIndex([])
order_right = pd.date_range(r_s, r_e, freq="D")
category_array = (
    ([f"실적|{d.strftime('%Y-%m-%d')}" for d in order_left]) +
    [f"예측|{d.strftime('%Y-%m-%d')}" for d in order_right]
)
if not df_sel.empty:
    df_sel["x_cat"] = df_sel.apply(lambda r: f"{'실적' if r['period']=='실적기간' else '예측'}|{r['date'].strftime('%Y-%m-%d')}", axis=1)

# ===================== 제목과 그래프 사이: 3개 요약 카드 =====================
# 2025년 실적 합계
year_start_2025 = pd.to_datetime("2025-01-01")
year_end_2025   = pd.to_datetime("2025-12-31")
actual_2025 = actual_df_all[(actual_df_all["date"] >= year_start_2025) & (actual_df_all["date"] <= ACT_END)]
sum_sales_2025_w = int(pd.to_numeric(actual_2025["sales_amount"], errors="coerce").fillna(0).sum())
sum_pax_2025     = int(pd.to_numeric(actual_2025["passengers"],   errors="coerce").fillna(0).sum())

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
        <div class="summary-card">
          <div class="summary-title">2025년 총 매출액 (실적)</div>
          <p class="summary-value">{sum_sales_2025_w:,.0f} 원</p>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
        <div class="summary-card">
          <div class="summary-title">2025년 총 승객수 (실적)</div>
          <p class="summary-value">{sum_pax_2025:,.0f} 명</p>
        </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
        <div class="summary-card">
          <div class="summary-title">예측 정확도</div>
          <p class="summary-value">— %</p>
        </div>
    """, unsafe_allow_html=True)

# =================== 그래프 패널 ===================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("예측그래프")

# --- 체크박스(우측 정렬) ---
if not left_df.empty:
    sp, c1cb, c2cb, c3cb, c4cb = st.columns([6,1.6,1.6,1.8,1.6])
    with c1cb:
        show_act_sales = st.checkbox("매출액(실적)", True, key="cb_act_sales")
        st.markdown('<span class="lg-line" title="실선(실적)"></span>', unsafe_allow_html=True)
    with c2cb:
        show_act_pax = st.checkbox("승객수(실적)", True, key="cb_act_pax")
        st.markdown('<span class="lg-bar" title="막대(실적)"></span>', unsafe_allow_html=True)
    with c3cb:
        show_fct_sales = st.checkbox("매출액(예측)", True, key="cb_fct_sales")
        st.markdown('<span class="lg-line-dash" title="점선(예측)"></span>', unsafe_allow_html=True)
    with c4cb:
        show_fct_pax = st.checkbox("승객수(예측)", True, key="cb_fct_pax")
        st.markdown('<span class="lg-bar-f" title="막대(예측)"></span>', unsafe_allow_html=True)
else:
    sp, c3cb, c4cb = st.columns([8,2,2])
    with c3cb:
        show_fct_sales = st.checkbox("매출액(예측)", True, key="cb_fct_sales_only")
        st.markdown('<span class="lg-line-dash" title="점선(예측)"></span>', unsafe_allow_html=True)
    with c4cb:
        show_fct_pax = st.checkbox("승객수(예측)", True, key="cb_fct_pax_only")
        st.markdown('<span class="lg-bar-f" title="막대(예측)"></span>', unsafe_allow_html=True)
    show_act_sales = False
    show_act_pax = False

fig = go.Figure()
color_sales = "#1f77b4"
color_pax   = "#ff7f0e"

# 배경 음영(실적/예측 영역)
shapes = []
if len(order_left) > 0:
    shapes.append(dict(type="rect", xref="x", yref="paper",
                       x0=category_array[0], x1=category_array[len(order_left)-1],
                       y0=0, y1=1, fillcolor=HILITE2, line=dict(width=0), layer="below"))
if len(order_right) > 0:
    x0 = category_array[len(order_left)]
    x1 = category_array[-1]
    shapes.append(dict(type="rect", xref="x", yref="paper",
                       x0=x0, x1=x1, y0=0, y1=1,
                       fillcolor=HILITE1, line=dict(width=0), layer="below"))

# 승객 막대
if show_act_pax and not df_sel.empty:
    act_plot = df_sel[df_sel["source"].eq("actual")]
    if not act_plot.empty:
        fig.add_trace(go.Bar(
            x=act_plot["x_cat"], y=act_plot["passengers_k"],
            name="승객수(실적, 천명)", marker=dict(color=color_pax, line=dict(width=0)),
            opacity=0.55, offsetgroup="actual", yaxis="y2",
            hovertemplate="<b>%{x}</b><br>승객수: %{y:,.0f} 천명<extra></extra>"
        ))
if show_fct_pax and not df_sel.empty:
    fct_plot = df_sel[df_sel["source"].eq("forecast")]
    if not fct_plot.empty:
        fig.add_trace(go.Bar(
            x=fct_plot["x_cat"], y=fct_plot["passengers_k"],
            name="승객수(예측, 천명)",
            marker=dict(color=color_pax, pattern=dict(shape="/", fgcolor="rgba(0,0,0,0.45)", solidity=0.40), line=dict(width=0)),
            opacity=0.38, offsetgroup="forecast", yaxis="y2",
            hovertemplate="<b>%{x}</b><br>승객수(예측): %{y:,.0f} 천명<extra></extra>"
        ))

# 매출 선 (실적/예측 분리)
act_sales = df_sel[df_sel["source"].eq("actual")]
if show_act_sales and not act_sales.empty:
    fig.add_trace(go.Scatter(
        x=act_sales["x_cat"], y=act_sales["sales_million"],
        name="매출액(실적, 백만원)", mode="lines+markers",
        line=dict(color=color_sales, width=2.6, dash="solid"),
        marker=dict(size=6, color=color_sales),
        yaxis="y1", connectgaps=True,
        hovertemplate="<b>%{x}</b><br>매출액: %{y:,.0f} 백만원<extra></extra>"
    ))

fct_sales = df_sel[df_sel["source"].eq("forecast")]
if show_fct_sales and not fct_sales.empty:
    fig.add_trace(go.Scatter(
        x=fct_sales["x_cat"], y=fct_sales["sales_million"],
        name="매출액(예측, 백만원)", mode="lines",
        line=dict(color=color_sales, width=3.5, dash="dashdot"),
        yaxis="y1", connectgaps=True, hoverinfo="skip"
    ))

# 실적-예측 연결 보조선
if show_act_sales and show_fct_sales and (not act_sales.empty) and (not fct_sales.empty):
    last_act_row = act_sales.sort_values("date").iloc[-1]
    first_fct_row = fct_sales.sort_values("date").iloc[0]
    if pd.notna(last_act_row["sales_million"]) and pd.notna(first_fct_row["sales_million"]):
        fig.add_trace(go.Scatter(
            x=[last_act_row["x_cat"], first_fct_row["x_cat"]],
            y=[last_act_row["sales_million"], first_fct_row["sales_million"]],
            mode="lines",
            line=dict(color=color_sales, width=2.0, dash="solid"),
            yaxis="y1", hoverinfo="skip", showlegend=False
        ))

# x축 tick: '실적|' '예측|' 제거
tickvals, ticktext = [], []
if len(category_array) > 0:
    step = max(1, len(category_array)//6)
    for i in range(0, len(category_array), step):
        tickvals.append(category_array[i]); ticktext.append(category_array[i].split("|")[1])
    if category_array[-1] not in tickvals:
        tickvals.append(category_array[-1]); ticktext.append(category_array[-1].split("|")[1])

# 라벨 위치
left_mid_idx  = len(order_left)//2 if len(order_left)>0 else None
right_mid_idx = len(order_right)//2 if len(order_right)>0 else None
left_mid_cat  = category_array[left_mid_idx] if left_mid_idx is not None else None
right_mid_cat = category_array[(len(order_left) + right_mid_idx)] if right_mid_idx is not None else None

fig.update_layout(
    template=PLOTLY_TEMPLATE,
    hovermode="x unified",
    barmode="group", bargap=0.15, bargroupgap=0.05,
    shapes=shapes,
    xaxis=dict(title="", type="category", categoryorder="array", categoryarray=category_array,
               tickangle=-45, tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True),
    yaxis=dict(title="매출액(백만원)", tickformat=",.0f", showgrid=True, zeroline=False),
    yaxis2=dict(title="승객수(천명)", overlaying="y", side="right", tickformat=",.0f", showgrid=False, zeroline=False),
    showlegend=False,
    margin=dict(t=24, r=50, b=60, l=70),
    font=dict(family="Nanum Gothic, Malgun Gothic, AppleGothic, Noto Sans KR, Sans-Serif", size=13, color=TEXT),
    annotations=[
        *([dict(x=left_mid_cat,  y=0.95, xref="x", yref="paper", text="실적", showarrow=False,
                font=dict(size=12, color=SUBTEXT), align="center")] if left_mid_cat else []),
        *([dict(x=right_mid_cat, y=0.95, xref="x", yref="paper", text="예측", showarrow=False,
                font=dict(size=12, color="#60A5FA" if DARK else "#1D4ED8"), align="center")] if right_mid_cat else []),
    ],
    paper_bgcolor=PANEL_BG,
    plot_bgcolor=PANEL_BG,
)

config = dict(
    displaylogo=False,
    toImageButtonOptions=dict(format="png", filename=f"dual_axis_blocks_{date.today()}", scale=2),
    modeBarButtonsToAdd=["hovercompare"]
)
st.plotly_chart(fig, use_container_width=True, config=config)

# 안내 캡션
if l_s is not None:
    st.caption(f"실적(좌): {l_s.date()} ~ {l_e.date()} · 예측(우): {r_s.date()} ~ {r_e.date()} · 길이 {N_days}일 (동일)")
else:
    st.caption(f"예측만 표시: {r_s.date()} ~ {r_e.date()} · 길이 {N_days}일")
st.markdown('</div>', unsafe_allow_html=True)  # 패널 끝

# ===================== 그래프 아래: 기간 요약 카드(실적/예측) =====================
def sum_period(df, label, col):
    return int(pd.to_numeric(df.loc[df["period"].eq(label), col], errors="coerce").fillna(0).sum())

left_sales_m   = sum_period(df_sel, "실적기간", "sales_million")
left_pax_k     = sum_period(df_sel, "실적기간", "passengers_k")
right_sales_m  = sum_period(df_sel, "예측기간", "sales_million")
right_pax_k    = sum_period(df_sel, "예측기간", "passengers_k")

def pct_change(new, old):
    return np.nan if (old is None or old == 0) else (new - old) / old * 100.0

sales_pct = pct_change(right_sales_m, left_sales_m if left_sales_m>0 else None)
pax_pct   = pct_change(right_pax_k, left_pax_k if left_pax_k>0 else None)

colA, colB = st.columns(2)
with colA:
    st.markdown(
        f"""
        <div class="summary-card">
          <div class="summary-title">실적 기간 합계 ({l_s.date()} ~ {l_e.date()})</div>
          <p class="summary-value">매출액: {left_sales_m:,.0f} 백만원 · 승객수: {left_pax_k:,.0f} 천명</p>
        </div>
        """
        if l_s is not None else
        f"""
        <div class="summary-card">
          <div class="summary-title">실적 기간 합계</div>
          <p class="summary-value">—</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with colB:
    def fmt_delta_html(val):
        if isinstance(val, float) and not np.isnan(val):
            arrow = "▲" if val >= 0 else "▼"
            cls   = "delta-up" if val >= 0 else "delta-down"
            return f' <span class="{cls}">({arrow}{abs(val):.1f}%)</span>'
        return ""
    st.markdown(
        f"""
        <div class="summary-card">
          <div class="summary-title">예측 기간 합계 ({r_s.date()} ~ {r_e.date()})</div>
          <p class="summary-value">
            매출액: {right_sales_m:,.0f} 백만원{fmt_delta_html(sales_pct)} ·
            승객수: {right_pax_k:,.0f} 천명{fmt_delta_html(pax_pct)}
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===================== 그래프 아래: 표(실적/예측) =====================
left_dates  = pd.date_range(l_s, l_e, freq="D") if l_s is not None else pd.DatetimeIndex([])
right_dates = pd.date_range(r_s, r_e, freq="D")

left_tbl  = pd.DataFrame({"pos": range(len(left_dates)),  "date": left_dates}) if len(left_dates)>0 else pd.DataFrame(columns=["pos","date"])
right_tbl = pd.DataFrame({"pos": range(len(right_dates)), "date": right_dates})

if not left_df.empty:
    l_vals = left_df.set_index("date").reindex(left_dates)
    left_tbl["sales_million"] = (pd.to_numeric(l_vals["sales_amount"], errors="coerce")/1_000_000).values
    left_tbl["passengers_k"]  = (pd.to_numeric(l_vals["passengers"],   errors="coerce")/1_000).values

r_vals = right_df.set_index("date").reindex(right_dates) if not right_df.empty else pd.DataFrame(index=right_dates)
right_tbl["sales_million"] = (pd.to_numeric(r_vals.get("sales_amount"), errors="coerce")/1_000_000).values
right_tbl["passengers_k"]  = (pd.to_numeric(r_vals.get("passengers"),   errors="coerce")/1_000).values

merged_tbl = pd.merge(right_tbl, left_tbl[["pos","sales_million","passengers_k"]] if not left_tbl.empty else pd.DataFrame(columns=["pos"]), on="pos", how="left", suffixes=("_f","_a"))
def safe_pct(new, old):
    return np.nan if (pd.isna(old) or old == 0) else (new - old) / old * 100.0
merged_tbl["sales_pct"] = [safe_pct(n, o) for n, o in zip(merged_tbl["sales_million_f"], merged_tbl.get("sales_million_a"))]
merged_tbl["pax_pct"]   = [safe_pct(n, o) for n, o in zip(merged_tbl["passengers_k_f"],  merged_tbl.get("passengers_k_a"))]

# 체크박스 상태 반영(표 칼럼)
actual_df_show = pd.DataFrame()
if not left_tbl.empty:
    actual_df_show = pd.DataFrame({
        ("실적","일자"): left_tbl["date"].dt.strftime("%Y-%m-%d"),
        ("실적","매출액(백만원)"): left_tbl["sales_million"].round(0).astype("Int64") if 'cb_act_sales' in st.session_state and st.session_state.get('cb_act_sales') else pd.NA,
        ("실적","승객수(천명)"):   left_tbl["passengers_k"].round(0).astype("Int64") if 'cb_act_pax'   in st.session_state and st.session_state.get('cb_act_pax')   else pd.NA,
    })

fcast_dict = {("예측","일자"): right_tbl["date"].dt.strftime("%Y-%m-%d")}
if ('cb_fct_sales' in st.session_state and st.session_state.get('cb_fct_sales')) or ('cb_fct_sales_only' in st.session_state and st.session_state.get('cb_fct_sales_only')):
    fcast_dict[("예측","매출액(백만원(Δ))")] = [
        f"{int(round(v if not pd.isna(v) else 0)):,.0f}" + (f" ({'▲' if (not pd.isna(p) and p>=0) else ('▼' if not pd.isna(p) else '')}{'' if pd.isna(p) else f'{abs(p):.1f}%'} )" if not pd.isna(p) else "")
        for v,p in zip(right_tbl["sales_million"], merged_tbl["sales_pct"])
    ]
if ('cb_fct_pax' in st.session_state and st.session_state.get('cb_fct_pax')) or ('cb_fct_pax_only' in st.session_state and st.session_state.get('cb_fct_pax_only')):
    fcast_dict[("예측","승객수(천명(Δ))")] = [
        f"{int(round(v if not pd.isna(v) else 0)):,.0f}" + (f" ({'▲' if (not pd.isna(p) and p>=0) else ('▼' if not pd.isna(p) else '')}{'' if pd.isna(p) else f'{abs(p):.1f}%'} )" if not pd.isna(p) else "")
        for v,p in zip(right_tbl["passengers_k"], merged_tbl["pax_pct"])
    ]
forecast_df_show = pd.DataFrame(fcast_dict)

if not actual_df_show.empty:
    table_df = pd.concat([actual_df_show, forecast_df_show], axis=1)
else:
    table_df = forecast_df_show.copy()

# 합계행
left_sales_m   = int(round(left_tbl["sales_million"].sum())) if "sales_million" in left_tbl.columns else 0
left_pax_k     = int(round(left_tbl["passengers_k"].sum()))   if "passengers_k"  in left_tbl.columns else 0
right_sales_m2 = int(round(right_tbl["sales_million"].sum())) if "sales_million" in right_tbl.columns else 0
right_pax_k2   = int(round(right_tbl["passengers_k"].sum()))  if "passengers_k"  in right_tbl.columns else 0

def top_delta_str(val, base):
    if base is None or base == 0 or val is None:
        return ""
    delta = (val - base) / base * 100.0
    arrow = "▲" if delta >= 0 else "▼"
    return f" ({arrow}{abs(delta):.1f}%)"

sum_row = {}
if not actual_df_show.empty:
    sum_row.update({
        ("실적","일자"): "합계",
        ("실적","매출액(백만원)"): left_sales_m if ('cb_act_sales' in st.session_state and st.session_state.get('cb_act_sales')) else "",
        ("실적","승객수(천명)"):   left_pax_k   if ('cb_act_pax'   in st.session_state and st.session_state.get('cb_act_pax'))   else "",
    })
sum_row[("예측","일자")] = "합계"
if (('cb_fct_sales' in st.session_state and st.session_state.get('cb_fct_sales')) or ('cb_fct_sales_only' in st.session_state and st.session_state.get('cb_fct_sales_only'))):
    sum_row[("예측","매출액(백만원(Δ))")] = f"{right_sales_m2:,.0f}{top_delta_str(right_sales_m2, left_sales_m if left_sales_m>0 else None)}"
if (('cb_fct_pax' in st.session_state and st.session_state.get('cb_fct_pax')) or ('cb_fct_pax_only' in st.session_state and st.session_state.get('cb_fct_pax_only'))):
    sum_row[("예측","승객수(천명(Δ))")]   = f"{right_pax_k2:,.0f}{top_delta_str(right_pax_k2, left_pax_k if left_pax_k>0 else None)}"

# 멀티컬럼 정렬
if len(table_df.columns) > 0:
    table_df.columns = pd.MultiIndex.from_tuples(table_df.columns)
sum_row_df = pd.DataFrame([sum_row])
if len(table_df.columns) > 0:
    sum_row_df = sum_row_df.reindex(columns=table_df.columns)
table_df = pd.concat([sum_row_df, table_df], ignore_index=True)

st.markdown("#### 📋 그래프 표시 데이터 요약")
st.dataframe(
    table_df,
    use_container_width=True,
    height=min(520, 120 + 28 * (len(table_df)))
)
