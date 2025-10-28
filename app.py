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
st.title("📈 외부요인 기반 철도수요예측 시스템")

# ---- 글로벌 스타일: 간격/카드/패널/범례 ----
st.markdown(
    """
    <style>
    .gap-xl { height: 16px; }
    .panel {
        border: 1px solid #E5E7EB;
        background: #F8FAFC;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: inset 0 0 0 9999px rgba(148,163,184,0.04);
        margin-bottom: 8px;
    }
    .panel-tight {
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        border-radius: 10px;
        padding: 10px 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 8px;
    }
    .card-slim, .summary-card {
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .summary-title { font-size: 13px; color:#6B7280; margin:0 0 4px 0; }
    .summary-value { font-weight: 800; font-size: 18px; margin:0; }
    .card-slim h4 { font-size: 15px; margin: 0 0 6px 0; }
    .metric { font-weight: 700; font-size: 17px; margin: 2px 0; }
    .delta-up   { color: #1D4ED8; font-weight: 700; }
    .delta-down { color: #DC2626; font-weight: 700; }
    .muted { color: #6B7280; font-size: 12px; margin-top: 4px; }

    /* 범례 스와치 */
    .lg-line   { height:2px; border-top: 4px solid #1f77b4; border-radius:2px; display:inline-block; width:20px; margin-right:6px; }
    .lg-line-dash { height:2px; border-top: 0; border-bottom: 0; display:inline-block; width:20px; margin-right:6px;
                    background: linear-gradient(90deg, #1f77b4 40%, rgba(0,0,0,0) 40%) repeat-x;
                    background-size: 8px 4px; background-position: 0 50%; }
    .lg-bar    { background:#ff7f0e; display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; }
    .lg-bar-f  { background:#ff7f0e; opacity:0.7; display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; }
    .lg-text   { font-size: 13px; color:#111827; vertical-align:middle; }

    .legend-row { display:flex; gap:18px; align-items:center; flex-wrap:wrap; justify-content: flex-end; }
    .legend-item { display:flex; gap:8px; align-items:center; }
    </style>
    <div class="gap-xl"></div>
    """,
    unsafe_allow_html=True,
)

# ===== 기간 정의 =====
ACT_START = pd.to_datetime("2020-08-01")
ACT_END   = pd.to_datetime("2025-08-31")
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
    # ✅ 실적은 무조건 ACT_END까지로 클립
    df = df[(df["date"] >= ACT_START) & (df["date"] <= ACT_END)]
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
    f["sales_amount"] = np.nan  # 매출액은 없음(그래프용)
    return f

# ================= 예측 매출 로더 (forecast_sales.csv / .cvs 폴백) =================
@st.cache_data(show_spinner=False)
def load_forecast_sales_df() -> pd.DataFrame:
    """forecast_sales.csv 우선 → forecast_sales.cvs 폴백"""
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

# ================= 실적 로더 (train_reservations_rows.csv) =================
@st.cache_data(show_spinner=False)
def load_actual_rows_df() -> pd.DataFrame:
    """rows에서 일자별 합계 생성"""
    df = load_df_from_repo_csv("train_reservations_rows.csv").copy()
    required = ["travel_date", "passengers", "sales_amount"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"'train_reservations_rows.csv'에 다음 컬럼이 필요합니다: {', '.join(miss)}")

    df["travel_date"] = pd.to_datetime(df["travel_date"], errors="coerce")
    df = df.dropna(subset=["travel_date"])
    for c in ["passengers", "sales_amount"]:
        df[c] = (
            df[c].astype(str)
                 .str.replace(",", "", regex=False)
                 .replace("nan", np.nan)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    daily = (
        df.assign(date=df["travel_date"].dt.floor("D"))
          .groupby("date", as_index=False)[["passengers", "sales_amount"]]
          .sum()
          .sort_values("date")
    )
    # ✅ 실적 일일도 ACT_END까지만
    daily = daily[(daily["date"] >= ACT_START) & (daily["date"] <= ACT_END)]
    return daily

# ================= 유틸: 기간 보정/동일 길이 =================
def ensure_in_range(s: pd.Timestamp, e: pd.Timestamp, lo: pd.Timestamp, hi: pd.Timestamp):
    s2 = max(s, lo); e2 = min(e, hi)
    if s2 > e2: s2, e2 = lo, lo
    return s2, e2

def align_last_year_same_weekday(r_s: pd.Timestamp, n_days: int):
    """전년도 동일(요일) 시작일 계산"""
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

# 예측 기간
default_right_start = date(2025, 9, 1)
default_right_end   = date(2025, 9, 7)
right_range = st.session_state.get("right_range", (default_right_start, default_right_end))

right_sel = st.sidebar.date_input(
    "① 예측 기간 (YYYY-MM-DD)",
    value=right_range,
    min_value=FCT_START.date(), max_value=FCT_END.date(),
    key="right_picker_sidebar"
)

# 실적 기간 모드
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

# 병합본(그래프용)
df_sel = pd.concat(
    ([left_df.assign(period="실적기간")] if not left_df.empty else []) +
    [right_df.assign(period="예측기간")],
    ignore_index=True
).sort_values("date") if (not right_df.empty or not left_df.empty) else pd.DataFrame(columns=["date","passengers","sales_amount","source","period"])

# 단위 변환
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

# ===================== (NEW) 제목과 그래프 사이: 3개 요약 카드 =====================
# 2025년 실적만 합계 (원/명)
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

# --- 체크박스: 우측 정렬 & 순서 고정 ---
if not left_df.empty:
    sp, c1, c2, c3, c4 = st.columns([6,1.2,1.2,1.4,1.2])  # 우측 정렬
    with c1:
        show_act_sales = st.checkbox("매출액(실적)", True, key="cb_act_sales")
        st.markdown('<span class="lg-line"></span>', unsafe_allow_html=True)
    with c2:
        show_act_pax = st.checkbox("승객수(실적)", True, key="cb_act_pax")
        st.markdown('<span class="lg-bar"></span>', unsafe_allow_html=True)
    with c3:
        show_fct_sales = st.checkbox("매출액(예측)", True, key="cb_fct_sales")
        st.markdown('<span class="lg-line-dash" title="점선(예측)"></span>', unsafe_allow_html=True)
    with c4:
        show_fct_pax = st.checkbox("승객수(예측)", True, key="cb_fct_pax")
        st.markdown('<span class="lg-bar-f"></span>', unsafe_allow_html=True)
else:
    sp, c3, c4 = st.columns([8,1.6,1.6])
    with c3:
        show_fct_sales = st.checkbox("매출액(예측)", True, key="cb_fct_sales_only")
        st.markdown('<span class="lg-line-dash" title="점선(예측)"></span>', unsafe_allow_html=True)
    with c4:
        show_fct_pax = st.checkbox("승객수(예측)", True, key="cb_fct_pax_only")
        st.markdown('<span class="lg-bar-f"></span>', unsafe_allow_html=True)
    show_act_sales = False
    show_act_pax = False

# --- 그래프 본체 ---
fig = go.Figure()
color_sales = "#1f77b4"
color_pax   = "#ff7f0e"

# 배경 음영(실적/예측 영역)
shapes = []
if len(order_left) > 0:
    shapes.append(dict(type="rect", xref="x", yref="paper",
                       x0=category_array[0], x1=category_array[len(order_left)-1],
                       y0=0, y1=1, fillcolor="rgba(100,116,139,0.06)", line=dict(width=0), layer="below"))
if len(order_right) > 0:
    x0 = category_array[len(order_left)]
    x1 = category_array[-1]
    shapes.append(dict(type="rect", xref="x", yref="paper",
                       x0=x0, x1=x1, y0=0, y1=1,
                       fillcolor="rgba(30,144,255,0.08)", line=dict(width=0), layer="below"))

# 승객 막대
if show_act_pax and not df_sel.empty:
    act_plot = df_sel[df_sel["source"].eq("actual")]
    if not act_plot.empty:
        fig.add_trace(go.Bar(
            x=act_plot["x_cat"], y=act_plot["passengers_k"],
            name="승객수(실적, 천명)", marker=dict(color=color_pax, line=dict(width=0)),
            opacity=0.55, offsetgroup="actual", yaxis="y2",
            hovertemplate="<b>%{x}</b><br>승객수: %{y:,.1f} 천명<extra></extra>"
        ))
if show_fct_pax and not df_sel.empty:
    fct_plot = df_sel[df_sel["source"].eq("forecast")]
    if not fct_plot.empty:
        fig.add_trace(go.Bar(
            x=fct_plot["x_cat"], y=fct_plot["passengers_k"],
            name="승객수(예측, 천명)",
            marker=dict(color=color_pax, pattern=dict(shape="/", fgcolor="rgba(0,0,0,0.45)", solidity=0.40), line=dict(width=0)),
            opacity=0.38, offsetgroup="forecast", yaxis="y2",
            hovertemplate="<b>%{x}</b><br>승객수(예측): %{y:,.1f} 천명<extra></extra>"
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
        hovertemplate="<b>%{x}</b><br>매출액: %{y:,.1f} 백만원<extra></extra>"
    ))

fct_sales = df_sel[df_sel["source"].eq("forecast")]
if show_fct_sales and not fct_sales.empty:
    fig.add_trace(go.Scatter(
        x=fct_sales["x_cat"], y=fct_sales["sales_million"],
        name="매출액(예측, 백만원)", mode="lines",
        line=dict(color=color_sales, width=3.5, dash="dashdot"),
        yaxis="y1", connectgaps=True, hoverinfo="skip"
    ))

# 🔗 (NEW) 실적-예측 매출 꺾은선 연결 보조선
if show_act_sales and show_fct_sales and (not act_sales.empty) and (not fct_sales.empty):
    last_act_row = act_sales.sort_values("date").iloc[-1]
    first_fct_row = fct_sales.sort_values("date").iloc[0]
    if pd.notna(last_act_row["sales_million"]) and pd.notna(first_fct_row["sales_million"]):
        fig.add_trace(go.Scatter(
            x=[last_act_row["x_cat"], first_fct_row["x_cat"]],
            y=[last_act_row["sales_million"], first_fct_row["sales_million"]],
            mode="lines",
            line=dict(color=color_sales, width=2.2, dash="solid"),
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
    template="plotly_white",
    hovermode="x unified",
    barmode="group", bargap=0.15, bargroupgap=0.05,
    shapes=shapes,
    xaxis=dict(title="", type="category", categoryorder="array", categoryarray=category_array,
               tickangle=-45, tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True),
    yaxis=dict(title="매출액(백만원)", tickformat=",.1f", showgrid=True, zeroline=False),
    yaxis2=dict(title="승객수(천명)", overlaying="y", side="right", tickformat=",.1f", showgrid=False, zeroline=False),
    showlegend=False,
    margin=dict(t=24, r=50, b=60, l=70),
    font=dict(family="Nanum Gothic, Malgun Gothic, AppleGothic, Noto Sans KR, Sans-Serif", size=13),
    annotations=[
        *([dict(x=left_mid_cat,  y=0.95, xref="x", yref="paper", text="실적", showarrow=False,
                font=dict(size=12, color="#475569"), align="center")] if left_mid_cat else []),
        *([dict(x=right_mid_cat, y=0.95, xref="x", yref="paper", text="예측", showarrow=False,
                font=dict(size=12, color="#1D4ED8"), align="center")] if right_mid_cat else []),
    ]
)

config = dict(
    displaylogo=False,
    toImageButtonOptions=dict(format="png", filename=f"dual_axis_blocks_{date.today()}", scale=2),
    modeBarButtonsToAdd=["hovercompare"]
)
st.plotly_chart(fig, use_container_width=True, config=config)

if l_s is not None:
    st.caption(f"실적(좌): {l_s.date()} ~ {l_e.date()} · 예측(우): {r_s.date()} ~ {r_e.date()} · 길이 {N_days}일 (동일)")
else:
    st.caption(f"예측만 표시: {r_s.date()} ~ {r_e.date()} · 길이 {N_days}일")
st.markdown('</div>', unsafe_allow_html=True)

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
st.title("📈 외부요인 기반 철도수요예측 시스템")

# ---- 글로벌 스타일: 간격/카드/패널/범례 ----
st.markdown(
    """
    <style>
    .gap-xl { height: 16px; }
    .panel {
        border: 1px solid #E5E7EB;
        background: #F8FAFC;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: inset 0 0 0 9999px rgba(148,163,184,0.04);
        margin-bottom: 8px;
    }
    .panel-tight {
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        border-radius: 10px;
        padding: 10px 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 8px;
    }
    .card-slim, .summary-card {
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .summary-title { font-size: 13px; color:#6B7280; margin:0 0 4px 0; }
    .summary-value { font-weight: 800; font-size: 18px; margin:0; }
    .card-slim h4 { font-size: 15px; margin: 0 0 6px 0; }
    .metric { font-weight: 700; font-size: 17px; margin: 2px 0; }
    .delta-up   { color: #1D4ED8; font-weight: 700; }
    .delta-down { color: #DC2626; font-weight: 700; }
    .muted { color: #6B7280; font-size: 12px; margin-top: 4px; }

    /* 범례 스와치 */
    .lg-line   { height:2px; border-top: 4px solid #1f77b4; border-radius:2px; display:inline-block; width:20px; margin-right:6px; }
    .lg-line-dash { height:2px; border-top: 0; border-bottom: 0; display:inline-block; width:20px; margin-right:6px;
                    background: linear-gradient(90deg, #1f77b4 40%, rgba(0,0,0,0) 40%) repeat-x;
                    background-size: 8px 4px; background-position: 0 50%; }
    .lg-bar    { background:#ff7f0e; display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; }
    .lg-bar-f  { background:#ff7f0e; opacity:0.7; display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; }
    .lg-text   { font-size: 13px; color:#111827; vertical-align:middle; }

    .legend-row { display:flex; gap:18px; align-items:center; flex-wrap:wrap; justify-content: flex-end; }
    .legend-item { display:flex; gap:8px; align-items:center; }
    </style>
    <div class="gap-xl"></div>
    """,
    unsafe_allow_html=True,
)

# ===== 기간 정의 =====
ACT_START = pd.to_datetime("2020-08-01")
ACT_END   = pd.to_datetime("2025-08-31")
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
    # ✅ 실적은 무조건 ACT_END까지로 클립
    df = df[(df["date"] >= ACT_START) & (df["date"] <= ACT_END)]
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
    f["sales_amount"] = np.nan  # 매출액은 없음(그래프용)
    return f

# ================= 예측 매출 로더 (forecast_sales.csv / .cvs 폴백) =================
@st.cache_data(show_spinner=False)
def load_forecast_sales_df() -> pd.DataFrame:
    """forecast_sales.csv 우선 → forecast_sales.cvs 폴백"""
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

# ================= 실적 로더 (train_reservations_rows.csv) =================
@st.cache_data(show_spinner=False)
def load_actual_rows_df() -> pd.DataFrame:
    """rows에서 일자별 합계 생성"""
    df = load_df_from_repo_csv("train_reservations_rows.csv").copy()
    required = ["travel_date", "passengers", "sales_amount"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"'train_reservations_rows.csv'에 다음 컬럼이 필요합니다: {', '.join(miss)}")

    df["travel_date"] = pd.to_datetime(df["travel_date"], errors="coerce")
    df = df.dropna(subset=["travel_date"])
    for c in ["passengers", "sales_amount"]:
        df[c] = (
            df[c].astype(str)
                 .str.replace(",", "", regex=False)
                 .replace("nan", np.nan)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    daily = (
        df.assign(date=df["travel_date"].dt.floor("D"))
          .groupby("date", as_index=False)[["passengers", "sales_amount"]]
          .sum()
          .sort_values("date")
    )
    # ✅ 실적 일일도 ACT_END까지만
    daily = daily[(daily["date"] >= ACT_START) & (daily["date"] <= ACT_END)]
    return daily

# ================= 유틸: 기간 보정/동일 길이 =================
def ensure_in_range(s: pd.Timestamp, e: pd.Timestamp, lo: pd.Timestamp, hi: pd.Timestamp):
    s2 = max(s, lo); e2 = min(e, hi)
    if s2 > e2: s2, e2 = lo, lo
    return s2, e2

def align_last_year_same_weekday(r_s: pd.Timestamp, n_days: int):
    """전년도 동일(요일) 시작일 계산"""
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

# 예측 기간
default_right_start = date(2025, 9, 1)
default_right_end   = date(2025, 9, 7)
right_range = st.session_state.get("right_range", (default_right_start, default_right_end))

right_sel = st.sidebar.date_input(
    "① 예측 기간 (YYYY-MM-DD)",
    value=right_range,
    min_value=FCT_START.date(), max_value=FCT_END.date(),
    key="right_picker_sidebar"
)

# 실적 기간 모드
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

# 병합본(그래프용)
df_sel = pd.concat(
    ([left_df.assign(period="실적기간")] if not left_df.empty else []) +
    [right_df.assign(period="예측기간")],
    ignore_index=True
).sort_values("date") if (not right_df.empty or not left_df.empty) else pd.DataFrame(columns=["date","passengers","sales_amount","source","period"])

# 단위 변환
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

# ===================== (NEW) 제목과 그래프 사이: 3개 요약 카드 =====================
# 2025년 실적만 합계 (원/명)
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

# --- 체크박스: 우측 정렬 & 순서 고정 ---
if not left_df.empty:
    sp, c1, c2, c3, c4 = st.columns([6,1.2,1.2,1.4,1.2])  # 우측 정렬
    with c1:
        show_act_sales = st.checkbox("매출액(실적)", True, key="cb_act_sales")
        st.markdown('<span class="lg-line"></span>', unsafe_allow_html=True)
    with c2:
        show_act_pax = st.checkbox("승객수(실적)", True, key="cb_act_pax")
        st.markdown('<span class="lg-bar"></span>', unsafe_allow_html=True)
    with c3:
        show_fct_sales = st.checkbox("매출액(예측)", True, key="cb_fct_sales")
        st.markdown('<span class="lg-line-dash" title="점선(예측)"></span>', unsafe_allow_html=True)
    with c4:
        show_fct_pax = st.checkbox("승객수(예측)", True, key="cb_fct_pax")
        st.markdown('<span class="lg-bar-f"></span>', unsafe_allow_html=True)
else:
    sp, c3, c4 = st.columns([8,1.6,1.6])
    with c3:
        show_fct_sales = st.checkbox("매출액(예측)", True, key="cb_fct_sales_only")
        st.markdown('<span class="lg-line-dash" title="점선(예측)"></span>', unsafe_allow_html=True)
    with c4:
        show_fct_pax = st.checkbox("승객수(예측)", True, key="cb_fct_pax_only")
        st.markdown('<span class="lg-bar-f"></span>', unsafe_allow_html=True)
    show_act_sales = False
    show_act_pax = False

# --- 그래프 본체 ---
fig = go.Figure()
color_sales = "#1f77b4"
color_pax   = "#ff7f0e"

# 배경 음영(실적/예측 영역)
shapes = []
if len(order_left) > 0:
    shapes.append(dict(type="rect", xref="x", yref="paper",
                       x0=category_array[0], x1=category_array[len(order_left)-1],
                       y0=0, y1=1, fillcolor="rgba(100,116,139,0.06)", line=dict(width=0), layer="below"))
if len(order_right) > 0:
    x0 = category_array[len(order_left)]
    x1 = category_array[-1]
    shapes.append(dict(type="rect", xref="x", yref="paper",
                       x0=x0, x1=x1, y0=0, y1=1,
                       fillcolor="rgba(30,144,255,0.08)", line=dict(width=0), layer="below"))

# 승객 막대
if show_act_pax and not df_sel.empty:
    act_plot = df_sel[df_sel["source"].eq("actual")]
    if not act_plot.empty:
        fig.add_trace(go.Bar(
            x=act_plot["x_cat"], y=act_plot["passengers_k"],
            name="승객수(실적, 천명)", marker=dict(color=color_pax, line=dict(width=0)),
            opacity=0.55, offsetgroup="actual", yaxis="y2",
            hovertemplate="<b>%{x}</b><br>승객수: %{y:,.1f} 천명<extra></extra>"
        ))
if show_fct_pax and not df_sel.empty:
    fct_plot = df_sel[df_sel["source"].eq("forecast")]
    if not fct_plot.empty:
        fig.add_trace(go.Bar(
            x=fct_plot["x_cat"], y=fct_plot["passengers_k"],
            name="승객수(예측, 천명)",
            marker=dict(color=color_pax, pattern=dict(shape="/", fgcolor="rgba(0,0,0,0.45)", solidity=0.40), line=dict(width=0)),
            opacity=0.38, offsetgroup="forecast", yaxis="y2",
            hovertemplate="<b>%{x}</b><br>승객수(예측): %{y:,.1f} 천명<extra></extra>"
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
        hovertemplate="<b>%{x}</b><br>매출액: %{y:,.1f} 백만원<extra></extra>"
    ))

fct_sales = df_sel[df_sel["source"].eq("forecast")]
if show_fct_sales and not fct_sales.empty:
    fig.add_trace(go.Scatter(
        x=fct_sales["x_cat"], y=fct_sales["sales_million"],
        name="매출액(예측, 백만원)", mode="lines",
        line=dict(color=color_sales, width=3.5, dash="dashdot"),
        yaxis="y1", connectgaps=True, hoverinfo="skip"
    ))

# 🔗 (NEW) 실적-예측 매출 꺾은선 연결 보조선
if show_act_sales and show_fct_sales and (not act_sales.empty) and (not fct_sales.empty):
    last_act_row = act_sales.sort_values("date").iloc[-1]
    first_fct_row = fct_sales.sort_values("date").iloc[0]
    if pd.notna(last_act_row["sales_million"]) and pd.notna(first_fct_row["sales_million"]):
        fig.add_trace(go.Scatter(
            x=[last_act_row["x_cat"], first_fct_row["x_cat"]],
            y=[last_act_row["sales_million"], first_fct_row["sales_million"]],
            mode="lines",
            line=dict(color=color_sales, width=2.2, dash="solid"),
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
    template="plotly_white",
    hovermode="x unified",
    barmode="group", bargap=0.15, bargroupgap=0.05,
    shapes=shapes,
    xaxis=dict(title="", type="category", categoryorder="array", categoryarray=category_array,
               tickangle=-45, tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True),
    yaxis=dict(title="매출액(백만원)", tickformat=",.1f", showgrid=True, zeroline=False),
    yaxis2=dict(title="승객수(천명)", overlaying="y", side="right", tickformat=",.1f", showgrid=False, zeroline=False),
    showlegend=False,
    margin=dict(t=24, r=50, b=60, l=70),
    font=dict(family="Nanum Gothic, Malgun Gothic, AppleGothic, Noto Sans KR, Sans-Serif", size=13),
    annotations=[
        *([dict(x=left_mid_cat,  y=0.95, xref="x", yref="paper", text="실적", showarrow=False,
                font=dict(size=12, color="#475569"), align="center")] if left_mid_cat else []),
        *([dict(x=right_mid_cat, y=0.95, xref="x", yref="paper", text="예측", showarrow=False,
                font=dict(size=12, color="#1D4ED8"), align="center")] if right_mid_cat else []),
    ]
)

config = dict(
    displaylogo=False,
    toImageButtonOptions=dict(format="png", filename=f"dual_axis_blocks_{date.today()}", scale=2),
    modeBarButtonsToAdd=["hovercompare"]
)
st.plotly_chart(fig, use_container_width=True, config=config)

if l_s is not None:
    st.caption(f"실적(좌): {l_s.date()} ~ {l_e.date()} · 예측(우): {r_s.date()} ~ {r_e.date()} · 길이 {N_days}일 (동일)")
else:
    st.caption(f"예측만 표시: {r_s.date()} ~ {r_e.date()} · 길이 {N_days}일")
st.markdown('</div>', unsafe_allow_html=True)

# =================== 그래프 하단: "그래프에 보이는 데이터" 표 (정수 표기) ===================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("##### 📋 그래프 표시 데이터 (정수)")

def daily_block(df_in: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    days = pd.date_range(start, end, freq="D")
    base = pd.DataFrame({"date": days})
    if df_in.empty:
        use = pd.DataFrame({"date": days, "sales_amount": np.nan, "passengers": np.nan})
    else:
        use = df_in.copy()
    use["sales_amount"] = pd.to_numeric(use["sales_amount"], errors="coerce")
    use["passengers"]   = pd.to_numeric(use["passengers"], errors="coerce")
    out = base.merge(use[["date","sales_amount","passengers"]], on="date", how="left")
    # 단위 변환 → 정수 반올림
    out["매출액(백만원)"] = np.rint(out["sales_amount"] / 1_000_000).astype("Int64")
    out["승객수(천명)"]   = np.rint(out["passengers"] / 1_000).astype("Int64")
    out["일자"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[["일자","매출액(백만원)","승객수(천명)"]]

left_block  = daily_block(left_df,  l_s, l_e) if l_s is not None else pd.DataFrame(columns=["일자","매출액(백만원)","승객수(천명)"])
right_block = daily_block(right_df, r_s, r_e)

# 행 단위 증감율(실적 대비)
if not left_block.empty:
    sales_base = left_block["매출액(백만원)"].astype("Float64")
    pax_base   = left_block["승객수(천명)"].astype("Float64")
    sales_pct_row = (right_block["매출액(백만원)"].astype("Float64") - sales_base) / sales_base * 100
    pax_pct_row   = (right_block["승객수(천명)"].astype("Float64") - pax_base)   / pax_base   * 100
else:
    sales_pct_row = pd.Series([np.nan]*len(right_block))
    pax_pct_row   = pd.Series([np.nan]*len(right_block))

def fmt_int_with_delta(val, pct):
    if pd.isna(val):
        return "-"
    base = f"{int(val):,}"
    if pd.isna(pct):
        return base
    arrow = "▲" if pct >= 0 else "▽"
    return f"{base} ({arrow}{abs(pct):.1f}%)"

right_block_disp = pd.DataFrame({
    "일자": right_block["일자"],
    "매출액(▲증감율%)": [fmt_int_with_delta(v, p) for v, p in zip(right_block["매출액(백만원)"], sales_pct_row)],
    "승객수(▲증감율%)": [fmt_int_with_delta(v, p) for v, p in zip(right_block["승객수(천명)"], pax_pct_row)],
})

# 체크박스 상태에 맞춘 컬럼 구성
columns_plan = []
frames = []

if not left_block.empty:
    parts = {"(실적)일자": left_block["일자"]}
    if show_act_sales: parts["(실적)매출액(백만원)"] = left_block["매출액(백만원)"]
    if show_act_pax:   parts["(실적)승객수(천명)"]   = left_block["승객수(천명)"]
    frames.append(pd.DataFrame(parts))
    if "(실적)일자" not in columns_plan: columns_plan += [("실적","일자")]
    if show_act_sales: columns_plan += [("실적","매출액(백만원)")]
    if show_act_pax:   columns_plan += [("실적","승객수(천명)")]

parts_f = {"(예측)일자": right_block_disp["일자"]}
if show_fct_sales: parts_f["(예측)매출액(▲증감율%)"] = right_block_disp["매출액(▲증감율%)"]
if show_fct_pax:   parts_f["(예측)승객수(▲증감율%)"] = right_block_disp["승객수(▲증감율%)"]
frames.append(pd.DataFrame(parts_f))
columns_plan += [("예측","일자")]
if show_fct_sales: columns_plan += [("예측","매출액(▲증감율%)")]
if show_fct_pax:   columns_plan += [("예측","승객수(▲증감율%)")]

table = pd.concat(frames, axis=1)

# 컬럼 순서 확정
col_names = []
for top, sub in columns_plan:
    if top=="실적":
        if sub=="일자": col_names.append("(실적)일자")
        elif sub=="매출액(백만원)": col_names.append("(실적)매출액(백만원)")
        elif sub=="승객수(천명)":   col_names.append("(실적)승객수(천명)")
    else:
        if sub=="일자": col_names.append("(예측)일자")
        elif sub=="매출액(▲증감율%)": col_names.append("(예측)매출액(▲증감율%)")
        elif sub=="승객수(▲증감율%)": col_names.append("(예측)승객수(▲증감율%)")
table = table[col_names]

# 합계 행 (정수)
sum_row = {}
if "(실적)일자" in table.columns: sum_row["(실적)일자"] = "합계"
if "(실적)매출액(백만원)" in table.columns:
    sum_row["(실적)매출액(백만원)"] = f"{int(pd.to_numeric(left_block['매출액(백만원)'], errors='coerce').fillna(0).sum()):,}"
if "(실적)승객수(천명)" in table.columns:
    sum_row["(실적)승객수(천명)"]   = f"{int(pd.to_numeric(left_block['승객수(천명)'], errors='coerce').fillna(0).sum()):,}"

if "(예측)일자" in table.columns: sum_row["(예측)일자"] = "합계"
if "(예측)매출액(▲증감율%)" in table.columns:
    total_f_sales = int(pd.to_numeric(right_block["매출액(백만원)"], errors="coerce").fillna(0).sum())
    total_a_sales = pd.to_numeric(left_block["매출액(백만원)"], errors="coerce").fillna(0).sum() if not left_block.empty else np.nan
    pct = np.nan if (pd.isna(total_a_sales) or total_a_sales==0) else (total_f_sales-total_a_sales)/total_a_sales*100
    arrow = "" if pd.isna(pct) else ("▲" if pct>=0 else "▽")
    pct_txt = "" if pd.isna(pct) else f" ({arrow}{abs(pct):.1f}%)"
    sum_row["(예측)매출액(▲증감율%)"] = f"{total_f_sales:,}{pct_txt}"
if "(예측)승객수(▲증감율%)" in table.columns:
    total_f_pax = int(pd.to_numeric(right_block["승객수(천명)"], errors="coerce").fillna(0).sum())
    total_a_pax = pd.to_numeric(left_block["승객수(천명)"], errors="coerce").fillna(0).sum() if not left_block.empty else np.nan
    pct = np.nan if (pd.isna(total_a_pax) or total_a_pax==0) else (total_f_pax-total_a_pax)/total_a_pax*100
    arrow = "" if pd.isna(pct) else ("▲" if pct>=0 else "▽")
    pct_txt = "" if pd.isna(pct) else f" ({arrow}{abs(pct):.1f}%)"
    sum_row["(예측)승객수(▲증감율%)"] = f"{total_f_pax:,}{pct_txt}"

if len(sum_row) > 0:
    sum_df = pd.DataFrame([sum_row])
    for c in table.columns:
        if c not in sum_df.columns:
            sum_df[c] = ""
    sum_df = sum_df[table.columns]
    table_display = pd.concat([sum_df, table], ignore_index=True)
else:
    table_display = table.copy()

st.dataframe(table_display, use_container_width=True, height=420)
st.markdown('</div>', unsafe_allow_html=True)



