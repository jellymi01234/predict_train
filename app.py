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

# ---- 글로벌 스타일: 간격/카드/패널/배지 ----
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
    .card-slim {
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .card-slim h4 { font-size: 15px; margin: 0 0 6px 0; }
    .metric { font-weight: 700; font-size: 17px; margin: 2px 0; }
    .delta-up   { color: #1D4ED8; font-weight: 700; }  /* 파란색(상승) */
    .delta-down { color: #DC2626; font-weight: 700; }  /* 빨간색(하락) */
    .muted { color: #6B7280; font-size: 12px; margin-top: 4px; }

    /* 범례 스와치 */
    .lg-swatch { display:inline-block; width:14px; height:6px; margin-right:6px; border-radius:2px; }
    .lg-line   { height:2px; border-top: 4px solid #1f77b4; border-radius:2px; display:inline-block; width:18px; margin-right:6px; }
    .lg-bar    { background:#ff7f0e; display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
    .lg-text   { font-size: 13px; color:#111827; vertical-align:middle; }

    .legend-row { display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
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
    return daily

# ================= 유틸: 기간 보정/동일 길이 =================
def ensure_in_range(s: pd.Timestamp, e: pd.Timestamp, lo: pd.Timestamp, hi: pd.Timestamp):
    s2 = max(s, lo); e2 = min(e, hi)
    if s2 > e2: s2, e2 = lo, lo
    return s2, e2

def align_last_year_same_weekday(r_s: pd.Timestamp, n_days: int):
    """전년도 동일(요일) 시작일 계산 → 시작 요일을 맞춘 후 n일 길이 확보"""
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

df_sel = pd.concat(
    ([left_df.assign(period="실적기간")] if not left_df.empty else []) +
    [right_df.assign(period="예측기간")],
    ignore_index=True
).sort_values("date") if (not right_df.empty or not left_df.empty) else pd.DataFrame(columns=["date","passengers","sales_amount","source","period"])

# 예측 매출 주입
if not df_sel.empty:
    df_sel = df_sel.merge(forecast_sales_all, on="date", how="left")
    df_sel["sales_amount"] = np.where(
        df_sel["source"].eq("forecast") & df_sel["sales_amount"].isna(),
        df_sel["pred_sales_amount"],
        df_sel["sales_amount"]
    )

# 단위 변환
df_sel["sales_million"] = pd.to_numeric(df_sel["sales_amount"], errors="coerce") / 1_000_000
df_sel["passengers_k"]  = pd.to_numeric(df_sel["passengers"], errors="coerce") / 1_000  # 천명

# ================= X축(두 블록 카테고리) + 표시용 텍스트 =================
order_left  = pd.date_range(l_s, l_e, freq="D") if l_s is not None else pd.DatetimeIndex([])
order_right = pd.date_range(r_s, r_e, freq="D")
category_array = (
    ([f"실적|{d.strftime('%Y-%m-%d')}" for d in order_left]) +
    [f"예측|{d.strftime('%Y-%m-%d')}" for d in order_right]
)
def to_xcat(row):
    prefix = "실적" if row["period"] == "실적기간" else "예측"
    return f"{prefix}|{row['date'].strftime('%Y-%m-%d')}"
if not df_sel.empty:
    df_sel["x_cat"] = df_sel.apply(to_xcat, axis=1)

# =================== 그래프 패널(테두리/음영 포함) ===================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("예측그래프")

# === 표시할 항목 선택(제목과 그래프 사이) ===
# 실적 모드에 따라 동적 표시
if not left_df.empty:
    # 실적+예측 모두 존재 → 4개 항목
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_act_sales = st.checkbox("", value=True, key="cb_act_sales")
        st.markdown('<span class="lg-line"></span><span class="lg-text">매출액(실적, 백만원)</span>', unsafe_allow_html=True)
    with c2:
        show_act_pax = st.checkbox("", value=True, key="cb_act_pax")
        st.markdown('<span class="lg-bar"></span><span class="lg-text">승객수(실적, 천명)</span>', unsafe_allow_html=True)
    with c3:
        show_fct_sales = st.checkbox("", value=True, key="cb_fct_sales")
        st.markdown('<span class="lg-line" style="border-top-style:dashed;"></span><span class="lg-text">매출액(예측, 백만원)</span>', unsafe_allow_html=True)
    with c4:
        show_fct_pax = st.checkbox("", value=True, key="cb_fct_pax")
        st.markdown('<span class="lg-bar" style="opacity:0.7;"></span><span class="lg-text">승객수(예측, 천명)</span>', unsafe_allow_html=True)
else:
    # 예측만 존재 → 2개 항목
    c3, c4, _sp1, _sp2 = st.columns([1,1,1,1])
    with c3:
        show_fct_sales = st.checkbox("", value=True, key="cb_fct_sales_only")
        st.markdown('<span class="lg-line" style="border-top-style:dashed;"></span><span class="lg-text">매출액(예측, 백만원)</span>', unsafe_allow_html=True)
    with c4:
        show_fct_pax = st.checkbox("", value=True, key="cb_fct_pax_only")
        st.markdown('<span class="lg-bar" style="opacity:0.7;"></span><span class="lg-text">승객수(예측, 천명)</span>', unsafe_allow_html=True)
    # 실적 항목은 강제로 False
    show_act_sales = False
    show_act_pax = False

# === 그래프 본체 ===
fig = go.Figure()
color_sales = "#1f77b4"  # 매출(선)
color_pax   = "#ff7f0e"  # 승객(막대)

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

# 승객(막대)
if not df_sel.empty and show_act_pax:
    act_plot = df_sel[df_sel["source"].eq("actual")]
    if not act_plot.empty:
        fig.add_trace(go.Bar(
            x=act_plot["x_cat"], y=act_plot["passengers_k"],
            name="승객수(실적, 천명)",
            marker=dict(color=color_pax, line=dict(width=0)),
            opacity=0.55, offsetgroup="actual", yaxis="y2",
            hovertemplate="<b>%{x}</b><br>승객수: %{y:,.1f} 천명<extra></extra>"
        ))
if not df_sel.empty and show_fct_pax:
    fct_plot = df_sel[df_sel["source"].eq("forecast")]
    if not fct_plot.empty:
        fig.add_trace(go.Bar(
            x=fct_plot["x_cat"], y=fct_plot["passengers_k"],
            name="승객수(예측, 천명)",
            marker=dict(
                color=color_pax,
                pattern=dict(shape="/", fgcolor="rgba(0,0,0,0.45)", solidity=0.40),
                line=dict(width=0)
            ),
            opacity=0.38, offsetgroup="forecast", yaxis="y2",
            hovertemplate="<b>%{x}</b><br>승객수(예측): %{y:,.1f} 천명<extra></extra>"
        ))

# 매출(선)
if not df_sel.empty and show_act_sales:
    fig.add_trace(go.Scatter(
        x=df_sel["x_cat"], y=df_sel["sales_million"],
        name="매출액(실적, 백만원)", mode="lines+markers",
        line=dict(color=color_sales, width=2.6, dash="solid"),
        marker=dict(size=6, color=color_sales),
        yaxis="y1", connectgaps=True,
        hovertemplate="<b>%{x}</b><br>매출액: %{y:,.1f} 백만원<extra></extra>"
    ))
if not df_sel.empty and show_fct_sales:
    sales_million_forecast_only = np.where(df_sel["source"].eq("forecast"), df_sel["sales_million"], None)
    fig.add_trace(go.Scatter(
        x=df_sel["x_cat"], y=sales_million_forecast_only,
        name="매출액(예측, 백만원)", mode="lines",
        line=dict(color=color_sales, width=3.5, dash="dashdot"),
        yaxis="y1", connectgaps=True, hoverinfo="skip"
    ))

# x축 tick: '실적|' '예측|' 제거
tickvals, ticktext = [], []
if len(category_array) > 0:
    step = max(1, len(category_array)//6)
    for i in range(0, len(category_array), step):
        tickvals.append(category_array[i])
        ticktext.append(category_array[i].split("|")[1])
    if category_array[-1] not in tickvals:
        tickvals.append(category_array[-1])
        ticktext.append(category_array[-1].split("|")[1])

# 중간 라벨
left_mid_idx  = len(order_left)//2 if len(order_left)>0 else None
right_mid_idx = len(order_right)//2 if len(order_right)>0 else None
left_mid_cat  = category_array[left_mid_idx] if left_mid_idx is not None else None
right_mid_cat = category_array[(len(order_left) + right_mid_idx)] if right_mid_idx is not None else None

fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    barmode="group", bargap=0.15, bargroupgap=0.05,
    shapes=shapes,
    xaxis=dict(
        title="",
        type="category",
        categoryorder="array", categoryarray=category_array,
        tickangle=-45, tickmode="array",
        tickvals=tickvals, ticktext=ticktext,
        showgrid=True
    ),
    yaxis=dict(title="매출액(백만원)", tickformat=",.1f", showgrid=True, zeroline=False),
    yaxis2=dict(title="승객수(천명)", overlaying="y", side="right", tickformat=",.1f", showgrid=False, zeroline=False),
    showlegend=False,  # 내부 범례 숨김(외부 체크박스 사용)
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

# =================== 요약 카드(상시 표시) ===================
def agg_sum(df, period_label, col):
    return pd.to_numeric(df.loc[df["period"].eq(period_label), col], errors="coerce").fillna(0).sum()

left_sales_total_w  = int(agg_sum(df_sel, "실적기간", "sales_amount")) if not df_sel.empty else 0
left_pax_total      = int(agg_sum(df_sel, "실적기간", "passengers"))  if not df_sel.empty else 0
right_sales_total_w = int(agg_sum(df_sel, "예측기간", "sales_amount")) if not df_sel.empty else 0
right_pax_total     = int(agg_sum(df_sel, "예측기간", "passengers"))  if not df_sel.empty else 0

def pct_change(new, old):
    return np.nan if old == 0 else (new - old) / old * 100.0
sales_pct = pct_change(right_sales_total_w, left_sales_total_w) if left_sales_total_w>0 else np.nan
pax_pct   = pct_change(right_pax_total, left_pax_total) if left_pax_total>0 else np.nan

sales_delta = ""
pax_delta = ""
if not np.isnan(sales_pct):
    cls = "delta-up" if sales_pct >= 0 else "delta-down"
    arrow = "▲" if sales_pct >= 0 else "▽"
    sales_delta = f' <span class="{cls}">({arrow}{abs(sales_pct):.1f}%)</span>'
if not np.isnan(pax_pct):
    cls = "delta-up" if pax_pct >= 0 else "delta-down"
    arrow = "▲" if pax_pct >= 0 else "▽"
    pax_delta = f' <span class="{cls}">({arrow}{abs(pax_pct):.1f}%)</span>'

colA, colB = st.columns(2)
with colA:
    st.markdown(
        f"""
        <div class="card-slim">
          <h4>🟦 실적 {f'({l_s.date()} ~ {l_e.date()})' if l_s is not None else ''}</h4>
          <div class="metric">매출액 : 총 {left_sales_total_w/1_000_000:,.1f} 백만원</div>
          <div class="metric">승객수 : 총 {left_pax_total/1_000:,.1f} 천명</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with colB:
    st.markdown(
        f"""
        <div class="card-slim">
          <h4>🟧 예측 ({r_s.date()} ~ {r_e.date()})</h4>
          <div class="metric">매출액 : 총 {right_sales_total_w/1_000_000:,.1f} 백만원{sales_delta}</div>
          <div class="metric">승객수 : 총 {right_pax_total/1_000:,.1f} 천명{pax_delta}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =================== 일일 데이터(상시 표시) ===================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("##### 📅 일일 데이터 (블록별)")

dates_left  = pd.DataFrame({"date": pd.date_range(l_s, l_e, freq="D")}) if l_s is not None else pd.DataFrame(columns=["date"])
dates_right = pd.DataFrame({"date": pd.date_range(r_s, r_e, freq="D")})

# 실적 일일
try:
    act_daily_raw = load_actual_rows_df()
    act_daily = act_daily_raw[act_daily_raw["date"].between(l_s, l_e)] if l_s is not None else pd.DataFrame(columns=act_daily_raw.columns)
    act_daily = act_daily.copy()
    if not act_daily.empty:
        act_daily.rename(columns={"passengers":"act_passengers","sales_amount":"act_sales_amount"}, inplace=True)
        act_daily["act_sales_million"] = pd.to_numeric(act_daily["act_sales_amount"], errors="coerce") / 1_000_000
        act_daily["act_passengers_k"]  = pd.to_numeric(act_daily["act_passengers"], errors="coerce") / 1_000
        act_daily = act_daily[["date","act_passengers_k","act_sales_million"]]
except (FileNotFoundError, KeyError):
    act_daily = pd.DataFrame(columns=["date","act_passengers_k","act_sales_million"])

# 예측 일일
f_pass = (
    forecast_df_all[forecast_df_all["date"].between(r_s, r_e)]
    .loc[:, ["date", "passengers"]]
    .rename(columns={"passengers":"pred_passengers"})
)
try:
    f_sales_raw = load_forecast_sales_df()
    f_sales = f_sales_raw[f_sales_raw["date"].between(r_s, r_e)].copy()
    f_sales["pred_sales_million"] = pd.to_numeric(f_sales["pred_sales_amount"], errors="coerce") / 1_000_000
    f_sales = f_sales[["date","pred_sales_million"]]
except FileNotFoundError:
    f_sales = pd.DataFrame(columns=["date","pred_sales_million"])

left_table = (
    dates_left.merge(act_daily, on="date", how="left")
              .assign(구분="실적", 날짜=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
              .rename(columns={"act_passengers_k":"승객수(천명)","act_sales_million":"매출액(백만원)"})
              [["구분","날짜","승객수(천명)","매출액(백만원)"]]
    if not dates_left.empty else pd.DataFrame(columns=["구분","날짜","승객수(천명)","매출액(백만원)"])
)
right_table = (
    dates_right.merge(f_pass, on="date", how="left").merge(f_sales, on="date", how="left")
               .assign(구분="예측", 날짜=lambda d: d["date"].dt.strftime("%Y-%m-%d"),
                       승객천명=lambda d: pd.to_numeric(d["pred_passengers"], errors="coerce")/1_000)
               .rename(columns={"승객천명":"승객수(천명)","pred_sales_million":"매출액(백만원)"})
               [["구분","날짜","승객수(천명)","매출액(백만원)"]]
)
detail_df = pd.concat([left_table, right_table], ignore_index=True)

st.dataframe(
    detail_df.style.format({"승객수(천명)": "{:,.1f}", "매출액(백만원)": "{:,.1f}"}),
    use_container_width=True,
    height=360
)
st.markdown('</div>', unsafe_allow_html=True)
