# app.py ── Streamlit (https://<YOUR-APP>.streamlit.app)

import io
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import re

# ✅ Ag-Grid (합계행 상단 고정 & 컬럼 필터/정렬/선택 지원)
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
except Exception as _e:
    AgGrid = None

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

# ===================== 사이드바: 기간 선택 =====================
st.sidebar.markdown("---")
st.sidebar.subheader("📅 기간 선택")

default_right_start = date(2025, 9, 1)
default_right_end   = date(2025, 9, 7)
right_range = st.session_state.get("right_range", (default_right_start, default_right_end))
right_sel = st.sidebar.date_input("① 예측 기간 (YYYY-MM-DD)",
    value=right_range, min_value=FCT_START.date(), max_value=FCT_END.date(), key="right_picker_sidebar")

left_mode = st.sidebar.radio("② 실적 기간 모드",
    options=["사용 안 함 (예측만)", "전년도 동일(일자)", "전년도 동일(요일)", "사용자 지정"], index=1, key="left_mode_sidebar")

left_sel = None
if left_mode == "사용자 지정":
    left_range = st.session_state.get("left_range", (date(2024, 9, 1), date(2024, 9, 7)))
    left_sel = st.sidebar.date_input("실적 기간 (YYYY-MM-DD)",
        value=left_range, min_value=ACT_START.date(), max_value=ACT_END.date(), key="left_picker_sidebar")

# ================= 기간 정규화/동기화 =================
def norm_tuple(sel):
    return sel if isinstance(sel, tuple) else (sel, sel)

r_s, r_e = map(pd.to_datetime, norm_tuple(right_sel))
r_s, r_e = ensure_in_range(r_s, r_e, FCT_START, FCT_END)
N_days = (r_e - r_s).days + 1

if left_mode == "사용 안 함 (예측만)":
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

# =================== 그래프 패널 ===================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("예측그래프")

sp, cSales, cPax = st.columns([8,1.6,1.6])
with cSales: show_sales = st.checkbox("매출액", True, key="cb_sales")
with cPax:   show_pax   = st.checkbox("승객수", True, key="cb_pax")

st.markdown(
    """
    <div class="legend-row" style="margin-top:4px;">
      <div><span class="lg-line"></span><span class="lg-text">매출액(실적)</span></div>
      <div><span class="lg-line-dash"></span><span class="lg-text">매출액(예측)</span></div>
      <div><span class="lg-bar"></span><span class="lg-text">승객수(실적)</span></div>
      <div><span class="lg-bar-f"></span><span class="lg-text">승객수(예측)</span></div>
    </div>
    """, unsafe_allow_html=True
)

fig = go.Figure(); color_sales="#1f77b4"; color_pax="#ff7f0e"
shapes = []
if len(order_left)>0:
    shapes.append(dict(type="rect", xref="x", yref="paper", x0=category_array[0], x1=category_array[len(order_left)-1], y0=0, y1=1, fillcolor=HILITE2, line=dict(width=0), layer="below"))
if len(order_right)>0:
    shapes.append(dict(type="rect", xref="x", yref="paper", x0=category_array[len(order_left)], x1=category_array[-1], y0=0, y1=1, fillcolor=HILITE1, line=dict(width=0), layer="below"))

if show_pax and not df_sel.empty:
    act_plot = df_sel[df_sel["source"].eq("actual")]; fct_plot = df_sel[df_sel["source"].eq("forecast")]
    if not act_plot.empty:
        fig.add_trace(go.Bar(x=act_plot["x_cat"], y=act_plot["passengers_k"], name="승객수(실적)",
                             marker=dict(color=color_pax, line=dict(width=0)), opacity=0.55, offsetgroup="pax", yaxis="y2",
                             hovertemplate="<b>%{x}</b><br>승객수: %{y:,.0f} 천명<extra></extra>"))
    if not fct_plot.empty:
        fig.add_trace(go.Bar(x=fct_plot["x_cat"], y=fct_plot["passengers_k"], name="승객수(예측)",
                             marker=dict(color=color_pax, pattern=dict(shape="/", fgcolor="rgba(0,0,0,0.45)", solidity=0.40), line=dict(width=0)),
                             opacity=0.38, offsetgroup="pax", yaxis="y2",
                             hovertemplate="<b>%{x}</b><br>승객수(예측): %{y:,.0f} 천명<extra></extra>"))

if show_sales and not df_sel.empty:
    act_sales = df_sel[df_sel["source"].eq("actual")]; fct_sales = df_sel[df_sel["source"].eq("forecast")]
    if not act_sales.empty:
        fig.add_trace(go.Scatter(x=act_sales["x_cat"], y=act_sales["sales_million"], name="매출액(실적)", mode="lines+markers",
                                 line=dict(color=color_sales, width=2.6), marker=dict(size=6, color=color_sales),
                                 yaxis="y1", connectgaps=True,
                                 hovertemplate="<b>%{x}</b><br>매출액: %{y:,.0f} 백만원<extra></extra>"))
    if not fct_sales.empty:
        fig.add_trace(go.Scatter(x=fct_sales["x_cat"], y=fct_sales["sales_million"], name="매출액(예측)", mode="lines",
                                 line=dict(color=color_sales, width=3.5, dash="dash"),
                                 yaxis="y1", connectgaps=True, hoverinfo="skip"))
    if (not act_sales.empty) and (not fct_sales.empty):
        la = act_sales.sort_values("date").iloc[-1]; ff = fct_sales.sort_values("date").iloc[0]
        if pd.notna(la["sales_million"]) and pd.notna(ff["sales_million"]):
            fig.add_trace(go.Scatter(x=[la["x_cat"], ff["x_cat"]], y=[la["sales_million"], ff["sales_million"]],
                                     mode="lines", line=dict(color=color_sales, width=2.0), yaxis="y1",
                                     hoverinfo="skip", showlegend=False))

tickvals, ticktext = [], []
if len(category_array)>0:
    step = max(1, len(category_array)//6)
    for i in range(0, len(category_array), step):
        tickvals.append(category_array[i]); ticktext.append(category_array[i].split("|")[1])
    if category_array[-1] not in tickvals:
        tickvals.append(category_array[-1]); ticktext.append(category_array[-1].split("|")[1])

left_mid_idx = len(order_left)//2 if len(order_left)>0 else None
right_mid_idx= len(order_right)//2 if len(order_right)>0 else None
left_mid_cat = category_array[left_mid_idx] if left_mid_idx is not None else None
right_mid_cat= category_array[(len(order_left)+right_mid_idx)] if right_mid_idx is not None else None

fig.update_layout(template=PLOTLY_TEMPLATE, hovermode="x unified",
    barmode="group", bargap=0.15, bargroupgap=0.05, shapes=shapes,
    xaxis=dict(title="", type="category", categoryorder="array", categoryarray=category_array,
               tickangle=-45, tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True),
    yaxis=dict(title="매출액(백만원)", tickformat=",.0f", showgrid=True, zeroline=False),
    yaxis2=dict(title="승객수(천명)", overlaying="y", side="right", tickformat=",.0f", showgrid=False, zeroline=False),
    showlegend=False, margin=dict(t=24, r=50, b=60, l=70),
    font=dict(family="Nanum Gothic, Malgun Gothic, AppleGothic, Noto Sans KR, Sans-Serif", size=13, color=TEXT),
    annotations=[
        *([dict(x=left_mid_cat,  y=0.50, xref="x", yref="paper", text="실적", showarrow=False, font=dict(size=24, color="#000"), align="center")] if left_mid_cat else []),
        *([dict(x=right_mid_cat, y=0.50, xref="x", yref="paper", text="예측", showarrow=False, font=dict(size=24, color="#000"), align="center")] if right_mid_cat else []),
    ],
    paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG)

st.plotly_chart(fig, use_container_width=True, config=dict(displaylogo=False,
    toImageButtonOptions=dict(format="png", filename=f"dual_axis_blocks_{date.today()}", scale=2),
    modeBarButtonsToAdd=["hovercompare"]))

if l_s is not None:
    st.caption(f"실적(좌): {l_s.date()} ~ {l_e.date()} · 예측(우): {r_s.date()} ~ {r_e.date()} · 길이 {N_days}일 (동일)")
else:
    st.caption(f"예측만 표시: {r_s.date()} ~ {r_e.date()} · 길이 {N_days}일")

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
# ======== 매트릭스(생성 → 전치) : 행=일자, 열=지표 ========
# ======== 매트릭스(생성 → 전치) : 행=일자, 열=지표 ========
st.markdown("#### 📋 데이터 매트릭스 (행=일자, 열=지표)")

# ---- 실적 매트릭스 생성 ----
def _build_left_matrix() -> pd.DataFrame:
    if left_tbl.empty:
        return pd.DataFrame()
    rows = {}
    if st.session_state.get("cb_sales", True) and "sales_million" in left_tbl:
        rows["매출액(백만원)|실적"] = left_tbl["sales_million"].round(0).astype("Int64").tolist()
    if st.session_state.get("cb_pax", True) and "passengers_k" in left_tbl:
        rows["승객수(천명)|실적"] = left_tbl["passengers_k"].round(0).astype("Int64").tolist()

    df = pd.DataFrame.from_dict(rows, orient="index", columns=fmt_date_ko(left_tbl["date"]))

    # 합계(숫자 행만)
    sums = []
    for idx in df.index:
        s = pd.to_numeric(df.loc[idx], errors="coerce").sum(min_count=1)
        sums.append("" if pd.isna(s) else int(round(s)))
    df.insert(0, "합계", sums)
    return df

# ---- 예측 매트릭스 생성 ----
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
    # 합계
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

# ---- 전치 ----
def _transpose_with_sum_first(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    t = df.T
    if "합계" in t.index:
        t = pd.concat([t.loc[["합계"]], t.drop(index=["합계"])], axis=0)
    return t

left_T  = _transpose_with_sum_first(left_matrix)
right_T = _transpose_with_sum_first(right_matrix)

# ==== [실적기간(전치)] 컬럼 순서: '휴일' → '외부요인' 로 맨 끝에 정렬 + 숫자 3자리 콤마 ====
if not left_T.empty:
    # 1) 맨 오른쪽에 '휴일' 그다음 '외부요인' 순서로 재배치
    if "휴일" in left_T.columns:
        _col = left_T.pop("휴일")
        left_T["휴일"] = _col
    if "외부요인" in left_T.columns:
        _col = left_T.pop("외부요인")
        left_T["외부요인"] = _col

    # 2) 매출액/승객수는 3자리 콤마로 포맷 (합계 행 포함)
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



# ==== 실적기간(전치) 표에 '외부요인'과 '휴일' 컬럼 추가 ====
if not left_T.empty and not left_tbl.empty:
    # 외부요인 문자열 ("이벤트 N")
    ext_values = build_event_strings(pd.DatetimeIndex(left_tbl["date"]), external_factors_df)
    # 휴일 라벨 (6자 이내)
    left_holiday_labels, _ = build_holiday_labels(pd.DatetimeIndex(left_tbl["date"]), holidays_df, max_len=6)

    # 전치 테이블 인덱스와 맞춰서 컬럼 길이 정렬
    def _append_aligned_column(T: pd.DataFrame, dates: pd.Series, values: list, col_name: str):
        date_labels = list(fmt_date_ko(pd.Series(dates)))
        mapping = {lbl: val for lbl, val in zip(date_labels, values)}
        aligned = []
        for idx in T.index:
            aligned.append("" if str(idx) == "합계" else mapping.get(idx, ""))
        T[col_name] = aligned
        return T

    left_T = _append_aligned_column(left_T, left_tbl["date"], ext_values, "외부요인")
    left_T = _append_aligned_column(left_T, left_tbl["date"], left_holiday_labels, "휴일")

# ==== 예측기간(전치) 표: 맨 오른쪽에 '휴일'만 추가 ====
if not right_T.empty and not right_tbl.empty:
    # 도우미: 전치 테이블 인덱스(합계 포함)에 맞춰 안전하게 컬럼 추가
    def _append_aligned_column(T: pd.DataFrame, dates: pd.Series, values: list, col_name: str):
        date_labels = list(fmt_date_ko(pd.Series(dates)))
        mapping = {lbl: val for lbl, val in zip(date_labels, values)}
        aligned = []
        for idx in T.index:
            aligned.append("" if str(idx) == "합계" else mapping.get(idx, ""))
        T[col_name] = aligned
        return T

    # 1) 혹시 이전 코드로 '휴무'가 있었다면 제거
    if "휴무" in right_T.columns:
        right_T = right_T.drop(columns=["휴무"])

    # 2) 휴일 라벨 생성
    right_holiday_labels, _ = build_holiday_labels(pd.DatetimeIndex(right_tbl["date"]), holidays_df, max_len=6)

    # 3) 이미 '휴일'이 있으면 맨 뒤로 이동(pop→재할당), 없으면 새로 추가
    if "휴일" in right_T.columns:
        _col = right_T.pop("휴일")
        right_T["휴일"] = _col
    else:
        right_T = _append_aligned_column(right_T, right_tbl["date"], right_holiday_labels, "휴일")



# ---- (도우미) 전치 테이블의 인덱스에 맞춰 안전하게 컬럼 추가 ----
def _append_aligned_column(T: pd.DataFrame, dates: pd.Series, values: list, col_name: str):
    """
    T: 전치된 테이블 (index = ["합계", fmt_date_ko(...), ...])
    dates: 원본 날짜 Series (예: left_tbl["date"] or right_tbl["date"])
    values: 날짜 개수만큼의 값 리스트 (예: 외부요인/휴일 라벨들)
    col_name: 추가할 컬럼명
    """
    if T is None or T.empty:
        return T
    # 전치 테이블의 인덱스(일자 라벨)에 맞춰 매핑
    date_labels = list(fmt_date_ko(pd.Series(dates)))
    mapping = {lbl: val for lbl, val in zip(date_labels, values)}

    aligned = []
    for idx in T.index:
        if str(idx) == "합계":
            aligned.append("")  # 합계 행에는 빈값
        else:
            aligned.append(mapping.get(idx, ""))  # 해당 일자 없으면 빈값

# ---- 스타일 ----
def _style_weekend_rows(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    blue_text = "#1e90ff"; red_text = "#ef4444"
    sty = df.style.set_properties(**{"text-align":"center"}) \
                  .set_table_styles([{"selector":"th","props":"text-align:center;"}])
    if "합계" in df.index:
        sty = sty.set_properties(subset=(["합계"], df.columns),
                                 **{"font-weight":"bold","background-color": SUM_BG})
    for idx in df.index:
        if isinstance(idx, str) and "(토)" in idx:
            sty = sty.set_properties(subset=([idx], df.columns), **{"color": blue_text})
        if isinstance(idx, str) and "(일)" in idx:
            sty = sty.set_properties(subset=([idx], df.columns), **{"color": red_text})
    return sty

# ---- 출력 ----
c1, c2 = st.columns(2)
with c1:
    st.markdown("**실적 기간 (전치)**")
    if left_T.empty:
        st.info("실적 기간 데이터가 없습니다.")
    else:
        # 1) 인덱스('일자')를 컬럼으로 꺼내기
        left_T.index.name = "일자"
        left_edit = left_T.reset_index()

        # 2) '외부요인' 오른쪽에 체크박스 컬럼 추가 (없으면 맨 끝에 추가)
        insert_pos = left_edit.columns.get_loc("외부요인") + 1 if "외부요인" in left_edit.columns else len(left_edit.columns)
        if "선택" not in left_edit.columns:
            left_edit.insert(insert_pos, "선택", False)

        # 3) 합계 행은 체크해도 무시하도록 표시(시각적으로는 체크 가능하지만 처리에서 제외)
        #    ※ st.data_editor는 행 단위 비활성화가 없어, 후처리에서 제외합니다.
        #    필요하면 '합계' 행에 안내 텍스트를 덧붙여도 됩니다.

        # 4) 에디터 렌더 (체크박스 포함)
        edited_left = st.data_editor(
            left_edit,
            hide_index=True,
            use_container_width=True,
            height=min(520, 140 + 28 * max(3, len(left_edit))),
            column_config={
                "선택": st.column_config.CheckboxColumn(
                    "선택",
                    help="해당 일자의 이벤트를 선택합니다.",
                    default=False,
                ),
            },
            disabled=["일자"],  # 날짜는 수정 못 하도록
        )

        # 5) 선택 결과를 세션에 저장 (합계 행 제외)
        selected_mask = (edited_left.get("선택") == True) & (edited_left.get("일자") != "합계")
        st.session_state["selected_event_dates_from_matrix"] = edited_left.loc[selected_mask, "일자"].tolist()





        # 6) 선택 요약 표시 (원하면 이 값을 통합 상세 섹션과 연동 가능)
        if st.session_state["selected_event_dates_from_matrix"]:
            st.caption("✅ 선택된 일자: " + ", ".join(st.session_state["selected_event_dates_from_matrix"]))
        else:
            st.caption("선택된 일자가 없습니다.")

# ==== 체크된 날짜의 이벤트 세로 나열 ====
def _label_to_date(lbl: str) -> pd.Timestamp | None:
    # "YYYY-MM-DD (요일)" 형식에서 앞 10자리만 파싱
    try:
        s = str(lbl).strip()
        iso = s[:10]  # "YYYY-MM-DD"
        dt = pd.to_datetime(iso, errors="coerce")
        return None if pd.isna(dt) else dt.normalize()
    except Exception:
        return None

# 1) 선택된 '일자' 라벨 → 날짜로 변환
_selected_labels = st.session_state.get("selected_event_dates_from_matrix", []) or []
_selected_dates = [d for d in (_label_to_date(x) for x in _selected_labels) if d is not None]
_selected_dates = sorted(set(_selected_dates))

# 2) 통합 이벤트 맵 확보 (세션에 없으면 즉석 생성)
integrated_map = st.session_state.get("integrated_event_map", None)

def _build_integrated_map_for_range(s: pd.Timestamp, e: pd.Timestamp) -> dict:
    if s is None or e is None or s > e:
        return {}
    visible_left = pd.date_range(s, e, freq="D")

    # 콘서트
    concert_counts_df = load_concert_counts_df()
    concert_info_df   = load_concert_info_df()
    concert_map = build_concert_map_by_date(visible_left, concert_counts_df, concert_info_df)

    # 박람회
    expo_counts_df = load_expo_counts_df()
    coex_info_df   = load_expo_info_df("coex_events_rows.csv",   "Coex")
    kintex_info_df = load_expo_info_df("kintex_events_rows.csv", "Kintex")
    bexco_info_df  = load_expo_info_df("bexco_events_rows.csv",  "Bexco")
    coex_map   = build_event_titles_by_date(visible_left, expo_counts_df, coex_info_df,   "coex_events_count")
    kintex_map = build_event_titles_by_date(visible_left, expo_counts_df, kintex_info_df, "kintex_events_count")
    bexco_map  = build_event_titles_by_date(visible_left, expo_counts_df, bexco_info_df,  "bexco_events_count")

    # 스포츠
    sports_counts_df = load_sports_counts_df()
    baseball_df = load_baseball_schedule_df()
    kleague_df  = load_kleague_schedule_df()
    baseball_map = build_single_day_titles_by_date(visible_left, sports_counts_df, baseball_df, "games_baseball", info_date_col="date")
    kleague_map  = build_single_day_titles_by_date(visible_left, sports_counts_df, kleague_df,  "games_soccer",   info_date_col="date")

    # 합치기
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

if integrated_map is None:
    # 실적 기간(l_s~l_e)을 기준으로 즉석 생성 (상단에서 이미 l_s, l_e 계산됨)
    integrated_map = _build_integrated_map_for_range(l_s, l_e)
    st.session_state["integrated_event_map"] = integrated_map  # 캐시

# 3) 화면 출력
st.markdown("#### 🔎 선택한 일자 이벤트")
if not _selected_dates:
    st.info("체크한 날짜가 없습니다.")
else:
    for d in _selected_dates:
        # 날짜 라벨 (요일 포함)
        pretty = fmt_date_ko(pd.Series([d])).iloc[0]
        events = integrated_map.get(d, [])
        st.write(f"**{pretty}**")
        if events:
            for t in events:
                st.markdown(f"- {t}")
        else:
            st.markdown("- (표시할 이벤트가 없습니다)")


with c2:
    st.markdown("**예측 기간 (전치)**")
    if right_T.empty:
        st.info("예측 기간 데이터가 없습니다.")
    else:
        right_T.index.name = "일자"
        st.dataframe(_style_weekend_rows(right_T), use_container_width=True,
                     height=min(520, 140 + 28 * max(3, len(right_T))))



# ===================== 9월 예측 정확도 (실적 vs 예측) =====================
st.markdown("#### 🎯 예측 정확도 (실적 vs 예측)")


SEP_START = pd.to_datetime("2025-09-01")
SEP_END   = pd.to_datetime("2025-09-30")

@st.cache_data(show_spinner=False)
def load_actual_sep_df() -> pd.DataFrame:
    """
    actual_sep_rows.csv에서 2025-09-01 ~ 2025-09-30 일자별 실적을 로드.
    - 다양한 컬럼명/형식을 견고하게 처리
    - 날짜 컬럼 자동 추론: travel_date / date / 일자 / 날짜 / (연,월,일 조합)
    - 숫자 전처리: 쉼표/공백/통화기호 제거 후 numeric 캐스팅
    - 동일 일자 중복 합산
    """
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

    # 1) 컬럼명 표준화
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 2) 날짜 컬럼 추론
    date_col_candidates = [
        "travel_date","date","일자","날짜","ts","dt"
    ]
    ymd_candidates = [
        ("year","month","day"),
        ("yyyy","mm","dd"),
        ("y","m","d"),
        ("s_y","s_m","s_d"),  # 종종 쓰던 포맷 대응
    ]
    date_series = None
    for c in date_col_candidates:
        if c in df.columns:
            date_series = pd.to_datetime(df[c], errors="coerce")
            break
    if date_series is None:
        # 연/월/일 분리형 조합 시도
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

    # 3) 승객/매출 컬럼 추론
    def _pick_col(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        # 부분 일치(예: 'sales amount', '매출(원)')
        for c in df.columns:
            for key in candidates:
                if key in c:
                    return c
        return None

    pax_col = _pick_col(["passengers","pax","ridership","승객","승객수"])
    sales_col = _pick_col(["sales_amount","sales","revenue","amount","매출","매출액"])

    if pax_col is None and sales_col is None:
        st.warning("actual_sep_rows.csv에서 승객/매출 컬럼을 찾을 수 없습니다.")
        return pd.DataFrame(columns=["date","passengers","sales_amount"])

    # 4) 숫자 전처리 유틸
    def to_numeric_clean(s):
        if s is None:
            return pd.Series(dtype="float64")
        return (
            pd.Series(s, dtype="object")
            .astype(str)
            .str.replace(r"[,\s₩원$₩]", "", regex=True)
            .replace({"": np.nan, "nan": np.nan})
            .pipe(pd.to_numeric, errors="coerce")
        )

    if pax_col is not None:
        df["passengers"] = to_numeric_clean(df[pax_col])
    else:
        df["passengers"] = np.nan

    if sales_col is not None:
        df["sales_amount"] = to_numeric_clean(df[sales_col])
    else:
        df["sales_amount"] = np.nan

    # 5) 유효 로우만 남기고 일자별 합산
    df = df.dropna(subset=["date"])
    if df[["passengers","sales_amount"]].isna().all(axis=None):
        st.warning("actual_sep_rows.csv의 승객/매출 값이 모두 결측입니다.")
        return pd.DataFrame(columns=["date","passengers","sales_amount"])

    daily = (df.groupby("date", as_index=False)[["passengers","sales_amount"]]
               .sum(min_count=1)  # 전부 NaN이면 NaN 유지
               .sort_values("date"))

    # 6) 9월 기간 필터
    daily = daily[(daily["date"] >= SEP_START) & (daily["date"] <= SEP_END)]

    if daily.empty:
        st.info("actual_sep_rows.csv에서 2025-09 기간에 해당하는 데이터가 없습니다.")
        # 디버그 도움말
        with st.expander("🔎 디버그: 원본 날짜 분포 보기"):
            try:
                tmp = pd.to_datetime(raw.iloc[:,0], errors="coerce")
                st.write("첫 번째 컬럼을 날짜로 캐스팅한 예시(무관할 수 있음):", tmp.min(), "~", tmp.max())
            except Exception:
                st.write("원본 미리보기:", raw.head())
        return pd.DataFrame(columns=["date","passengers","sales_amount"])

    return daily[["date","passengers","sales_amount"]]



def _safe_pct_err(forecast, actual):
    if pd.isna(actual) or actual == 0:
        return np.nan
    return (forecast - actual) / actual * 100.0


# --- 데이터 준비: 9월 실적 / 예측
actual_sep = load_actual_sep_df()

# forecast_pass.csv (승객) + forecast_sales.csv (매출)에서 9월만
fcst_pass_sep = forecast_df_all[(forecast_df_all["date"] >= SEP_START) & (forecast_df_all["date"] <= SEP_END)][["date","passengers"]].rename(columns={"passengers":"f_passengers"})
fcst_sales_sep = forecast_sales_all[(forecast_sales_all["date"] >= SEP_START) & (forecast_sales_all["date"] <= SEP_END)][["date","pred_sales_amount"]].rename(columns={"pred_sales_amount":"f_sales_amount"})

fcst_sep = pd.merge(fcst_pass_sep, fcst_sales_sep, on="date", how="outer").sort_values("date")

# --- 예측 기간 연동 (r_s, r_e)
sel_start = max(r_s, SEP_START)
sel_end   = min(r_e, SEP_END)
cmp = pd.merge(fcst_sep, actual_sep, on="date", how="outer")
cmp = cmp[(cmp["date"] >= sel_start) & (cmp["date"] <= sel_end)].sort_values("date").reset_index(drop=True)

cmp["a_passengers"]   = pd.to_numeric(cmp.get("passengers"), errors="coerce")
cmp["a_sales_amount"] = pd.to_numeric(cmp.get("sales_amount"), errors="coerce")
cmp["f_passengers"]   = pd.to_numeric(cmp.get("f_passengers"), errors="coerce")
cmp["f_sales_amount"] = pd.to_numeric(cmp.get("f_sales_amount"), errors="coerce")

cmp["pax_err_pct"]   = [ _safe_pct_err(fp, ap) for fp, ap in zip(cmp["f_passengers"], cmp["a_passengers"]) ]
cmp["sales_err_pct"] = [ _safe_pct_err(fs, as_) for fs, as_ in zip(cmp["f_sales_amount"], cmp["a_sales_amount"]) ]

# --- 표 생성 (단위: 매출=백만원, 승객=천명)
disp = pd.DataFrame({
    "일자": fmt_date_ko(cmp["date"].dt.tz_localize(None)) if "date" in cmp.columns else pd.Series(dtype=str),
    "실적|매출액(백만원)":  (cmp["a_sales_amount"] / 1_000_000).round(0).astype("Int64"),
    "예측|매출액(백만원)":  (cmp["f_sales_amount"] / 1_000_000).round(0).astype("Int64"),
    "오차율|매출액(%)":   cmp["sales_err_pct"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else ""),
    "실적|승객수(천명)":    (cmp["a_passengers"]  / 1_000).round(0).astype("Int64"),
    "예측|승객수(천명)":    (cmp["f_passengers"]  / 1_000).round(0).astype("Int64"),
    "오차율|승객수(%)":   cmp["pax_err_pct"].map(lambda x: f"{x:.1f}" if not pd.isna(x) else ""),
})

# --- 합계행 (MAPE)
def _mape_pct(f, a):
    s = [abs((fv - av)/av)*100.0 for fv, av in zip(f, a) if (not pd.isna(av) and av!=0 and not pd.isna(fv))]
    return np.nan if len(s)==0 else float(np.mean(s))

sum_a_pax   = cmp["a_passengers"].sum(skipna=True)
sum_f_pax   = cmp["f_passengers"].sum(skipna=True)
sum_a_sales = cmp["a_sales_amount"].sum(skipna=True)
sum_f_sales = cmp["f_sales_amount"].sum(skipna=True)
mape_pax   = _mape_pct(cmp["f_passengers"], cmp["a_passengers"])
mape_sales = _mape_pct(cmp["f_sales_amount"], cmp["a_sales_amount"])

sum_row = pd.DataFrame([{
    "일자": "합계",
    "실적|매출액(백만원)": int(round(sum_a_sales/1_000_000)) if not pd.isna(sum_a_sales) else pd.NA,
    "예측|매출액(백만원)": int(round(sum_f_sales/1_000_000)) if not pd.isna(sum_f_sales) else pd.NA,
    "오차율|매출액(%)":    round(mape_sales, 1) if not pd.isna(mape_sales) else pd.NA,
    "실적|승객수(천명)":   int(round(sum_a_pax/1_000)) if not pd.isna(sum_a_pax) else pd.NA,
    "예측|승객수(천명)":   int(round(sum_f_pax/1_000)) if not pd.isna(sum_f_pax) else pd.NA,
    "오차율|승객수(%)":    round(mape_pax, 1) if not pd.isna(mape_pax) else pd.NA,
}])

disp_out = pd.concat([sum_row, disp], ignore_index=True)

# --- 주말은 글씨색으로만 표시
def _weekday_textcolor_only_df(_df: pd.DataFrame) -> pd.DataFrame:
    blue_text = "#1e90ff"
    red_text  = "#ef4444"
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

# --- 표시
st.dataframe(
    disp_out.style
        .set_properties(**{"text-align":"center"})
        .set_table_styles([{"selector":"th","props":"text-align:center;"}])
        .apply(_weekday_textcolor_only_df, axis=None),
    use_container_width=True,
    height=min(520, 120 + 28 * (len(disp_out)+1))
)



