# app.py ── Streamlit (https://<YOUR-APP>.streamlit.app)
import io
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ================= 기본 설정 =================
st.set_page_config(page_title="Passengers & Sales (Dual Axis)", layout="wide")
st.title("📈 외부요인 기반 철도수요예측 시스템")

# ================= 사이드바 옵션 =================
st.sidebar.header("🧰 옵션")
interpolate_missing = st.sidebar.checkbox("결측치 보간(선 끊김 방지)", value=False)
use_rolling = st.sidebar.checkbox("이동평균(스무딩)", value=False)
window = st.sidebar.slider("이동평균 윈도우(일)", 2, 14, 3, 1, disabled=not use_rolling)

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

# ===== 기간 정의 =====
ACT_START = pd.to_datetime("2020-08-01")
ACT_END   = pd.to_datetime("2025-08-31")
FCT_START = pd.to_datetime("2025-09-01")
FCT_END   = pd.to_datetime("2025-11-29")

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
    """
    forecast_sales.csv(우선) → forecast_sales.cvs(폴백)에서
    date, forecast_90d를 읽어 pred_sales_amount로 반환
    """
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
    """
    train_reservations_rows.csv에서 일자별 실적 집계
    - 입력 컬럼: travel_date, passengers, sales_amount
    - 쉼표 제거 후 float 변환
    - 일자별 합계 반환 (date, passengers, sales_amount)
    """
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

# ================= 기간 선택 (예측기간만) =================
data_min = FCT_START.date()
data_max = FCT_END.date()

start_date, end_date = st.session_state.get("selected_range", (data_min, data_max))
tl, tr = st.columns([0.78, 0.22])
with tl:
    st.subheader("예측그래프")
with tr:
    with st.popover("📅 기간 설정", use_container_width=True):
        sel = st.date_input("시작일 / 종료일 (예측기간만)",
                            value=(start_date, end_date),
                            min_value=data_min, max_value=data_max,
                            key="range_picker_v5")
        if isinstance(sel, tuple):
            sel_start, sel_end = sel
        else:
            sel_start, sel_end = sel, sel
        st.caption("선택기간 길이(N일)과 동일한 이전 N일 데이터를 함께 표시합니다.")
start_date, end_date = pd.to_datetime(sel_start), pd.to_datetime(sel_end)
start_date = max(start_date, FCT_START)
end_date   = min(end_date,   FCT_END)
if start_date > end_date:
    st.stop()
st.session_state["selected_range"] = (start_date.date(), end_date.date())

# ================= 이전기간 계산 =================
N_days = (end_date - start_date).days + 1
prev_end   = start_date - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=N_days - 1)

# ================= 데이터 로드 및 병합 =================
actual_df_all   = load_actual_df()
forecast_df_all = load_forecast_df()
# 예측 매출(없어도 앱이 죽지 않도록 보호)
try:
    forecast_sales_all = load_forecast_sales_df()
except FileNotFoundError as e:
    st.warning(f"예측 매출 파일을 찾을 수 없어 매출 예측선을 그리지 못할 수 있습니다: {e}")
    forecast_sales_all = pd.DataFrame(columns=["date", "pred_sales_amount"])

def get_union_range(s, e):
    parts = []
    if not (e < ACT_START or s > ACT_END):
        a = actual_df_all[(actual_df_all["date"] >= s) & (actual_df_all["date"] <= e)]
        a = a.assign(source="actual")
        parts.append(a)
    if not (e < FCT_START or s > FCT_END):
        f = forecast_df_all[(forecast_df_all["date"] >= s) & (forecast_df_all["date"] <= e)]
        f = f.assign(source="forecast")
        parts.append(f)
    if parts:
        return pd.concat(parts, ignore_index=True).sort_values("date")
    return pd.DataFrame(columns=["date", "passengers", "sales_amount", "source"])

prev_df = get_union_range(prev_start, prev_end)
curr_df = get_union_range(start_date, end_date)
df_sel = pd.concat([
    prev_df.assign(period="이전기간"),
    curr_df.assign(period="선택기간")
], ignore_index=True).sort_values("date")

# --- 예측 매출 주입: 예측 구간(source=='forecast')에 pred_sales를 채워 선이 끊기지 않게 함
df_sel = df_sel.merge(forecast_sales_all, on="date", how="left")
df_sel["sales_amount"] = np.where(
    df_sel["source"].eq("forecast") & df_sel["sales_amount"].isna(),
    df_sel["pred_sales_amount"],
    df_sel["sales_amount"]
)

# 전처리(보간/스무딩은 주입된 sales_amount에 대해 적용)
if interpolate_missing and not df_sel.empty:
    df_sel = (df_sel.set_index("date")
              .groupby("period", group_keys=False)
              .apply(lambda g: g[["passengers","sales_amount"]]
                     .resample("D").asfreq()
              ).reset_index())
    df_sel[["passengers","sales_amount"]] = df_sel[["passengers","sales_amount"]].interpolate(method="time")

if use_rolling and not df_sel.empty:
    df_sel[["passengers","sales_amount"]] = (
        df_sel.groupby("period")[["passengers","sales_amount"]]
              .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )

df_sel["sales_million"] = pd.to_numeric(df_sel["sales_amount"], errors="coerce") / 1_000_000

# ================= 그래프 =================
def intersect(a1, a2, b1, b2):
    s = max(pd.to_datetime(a1), pd.to_datetime(b1))
    e = min(pd.to_datetime(a2), pd.to_datetime(b2))
    return (s <= e), s, e

# 화면 범위와 예측 음영 교집합
view_start, view_end = df_sel["date"].min(), df_sel["date"].max()
has_fct, fct_s, fct_e = intersect(view_start, view_end, FCT_START, FCT_END)

# 마스크
act_mask = df_sel["source"].eq("actual")
fct_mask = df_sel["source"].eq("forecast")

# 중앙 좌표 (배경 텍스트 위치)
prev_mid = prev_start + (prev_end - prev_start) / 2
curr_mid = start_date + (end_date - start_date) / 2

# 스팬 라벨링
def label_for_span(s, e):
    has_act_span = not df_sel[(df_sel["date"].between(s, e)) & act_mask].empty
    has_fct_span = not df_sel[(df_sel["date"].between(s, e)) & fct_mask].empty
    if has_act_span and has_fct_span: return "혼합"
    if has_act_span: return "실적"
    if has_fct_span: return "예측"
    return ""

prev_label = label_for_span(prev_start, prev_end)
curr_label = label_for_span(start_date, end_date)   # 선택기간은 보통 "예측"

# === 배경: 예측 구간만 연파랑 음영, 텍스트는 규칙에 맞춰 표시 ===
shapes, annotations = [], []
if has_fct:
    shapes.append(dict(
        type="rect", xref="x", yref="paper",
        x0=fct_s, x1=fct_e, y0=0, y1=1,
        fillcolor="rgba(30,144,255,0.08)", line=dict(width=0), layer="below"
    ))

# 요청 규칙: '이전기간,혼합'은 '실적' 문구 / '선택기간,예측'은 '예측' 문구
prev_text = "실적" if prev_label in ("실적", "혼합") else "예측"
curr_text = "예측"

annotations.extend([
    dict(x=prev_mid, y=0.5, xref="x", yref="paper",
         text=prev_text, showarrow=False,
         font=dict(size=28, color="rgba(30,30,30,0.28)"), align="center"),
    dict(x=curr_mid, y=0.5, xref="x", yref="paper",
         text=curr_text, showarrow=False,
         font=dict(size=28, color="rgba(0,91,187,0.38)"), align="center"),
])
fig = go.Figure()
fig.update_layout(shapes=shapes, annotations=annotations)

# === 색상 ===
color_sales = "#1f77b4"       # 매출(선, y1)
color_pax   = "#ff7f0e"       # 승객(막대, y2)

# --- 막대 먼저 추가(실적 → 예측), 마지막에 꺾은선 추가 ---
# 승객수(실적): 막대 (overlay 얇아짐 방지 → group 모드 사용, 폭 자동 확보)
act_plot = df_sel[act_mask]
if not act_plot.empty:
    fig.add_trace(go.Bar(
        x=act_plot["date"], y=act_plot["passengers"],
        name="승객수(실적)",
        marker=dict(color=color_pax, line=dict(width=0)),
        opacity=0.55,
        offsetgroup="actual",      # ✅ group 모드에서 서로 다른 그룹
        yaxis="y2",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>승객수: %{y:,.0f} 명<extra></extra>"
    ))

# 승객수(예측): 막대 (실적보다 더 투명)
fct_plot = df_sel[fct_mask]
if not fct_plot.empty:
    fig.add_trace(go.Bar(
        x=fct_plot["date"], y=fct_plot["passengers"],
        name="승객수(예측)",
        marker=dict(
            color=color_pax,
            pattern=dict(shape="/", fgcolor="rgba(0,0,0,0.45)", solidity=0.25),
            line=dict(width=0)
        ),
        opacity=0.38,
        offsetgroup="forecast",    # ✅ group 모드에서 서로 다른 그룹
        yaxis="y2",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>승객수(예측): %{y:,.0f} 명<extra></extra>"
    ))

# === 매출액(백만원): 실선 + 예측만 점선 오버레이 (끊김 없이 이어짐)
# 1) 전체 구간 실선 (legend: 매출액(실적))
fig.add_trace(go.Scatter(
    x=df_sel["date"], y=df_sel["sales_million"],
    name="매출액(실적)",
    mode="lines+markers",
    line=dict(color=color_sales, width=2.6, dash="solid"),
    marker=dict(size=6, color=color_sales),
    yaxis="y1",
    connectgaps=True,
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>매출액: %{y:,.1f} 백만원<extra></extra>"
))

# 2) 예측 구간만 점선 오버레이 (legend: 매출액(예측))
sales_million_forecast_only = np.where(fct_mask.to_numpy(), df_sel["sales_million"], None)
fig.add_trace(go.Scatter(
    x=df_sel["date"], y=sales_million_forecast_only,
    name="매출액(예측)",                  # ✅ legend 포함
    mode="lines",
    line=dict(color=color_sales, width=3.0, dash="dashdot"),  # ✅ 더 눈에 띄는 점선
    yaxis="y1",
    connectgaps=True,
    hoverinfo="skip"
))

# === 레이아웃/축/범례 ===
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    # 🔑 핵심: overlay → group 으로 변경해 막대 폭 정상화
    barmode="group",
    bargap=0.15,                 # 막대 간 간격(작을수록 두꺼움)
    bargroupgap=0.05,            # 그룹 간 간격
    xaxis=dict(title="", showgrid=True, tickformat="%Y-%m-%d", tickangle=-45),
    yaxis=dict(title="매출액(백만원)", tickformat=",.1f", showgrid=True, zeroline=False),
    yaxis2=dict(title="승객수", overlaying="y", side="right", tickformat=",.0f", showgrid=False, zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(t=60, r=50, b=60, l=70),
    font=dict(family="Nanum Gothic, Malgun Gothic, AppleGothic, Noto Sans KR, Sans-Serif", size=13)
)

config = dict(
    displaylogo=False,
    toImageButtonOptions=dict(format="png", filename=f"dual_axis_{start_date.date()}_{end_date.date()}", scale=2),
    modeBarButtonsToAdd=["hovercompare"]
)
st.plotly_chart(fig, use_container_width=True, config=config)

# ================= 요약표 =================
def agg_period_sum(df: pd.DataFrame, period_label: str, col: str):
    return pd.to_numeric(
        df.loc[df["period"].eq(period_label), col],
        errors="coerce"
    ).fillna(0).sum()

# df_sel에는 이미 예측 매출이 주입되어 있음 (sales_amount = 실적 + 예측값 주입)
prev_sales_m = agg_period_sum(df_sel, "이전기간", "sales_amount") / 1_000_000
curr_sales_m = agg_period_sum(df_sel, "선택기간", "sales_amount") / 1_000_000
prev_pax     = agg_period_sum(df_sel, "이전기간", "passengers")
curr_pax     = agg_period_sum(df_sel, "선택기간", "passengers")

def pct_change(new, old):
    return np.nan if old == 0 else (new - old) / old * 100.0

sales_pct, pax_pct = pct_change(curr_sales_m, prev_sales_m), pct_change(curr_pax, prev_pax)

summary_df = pd.DataFrame({
    "이전기간 합계": [prev_sales_m, prev_pax],
    "선택기간 합계": [curr_sales_m, curr_pax],
    "증감율(%)": [sales_pct, pax_pct],
}, index=["매출액(백만원)", "승객수(명)"])

idx = pd.IndexSlice
def style_delta(v):
    if pd.isna(v):
        return ""
    return "color: green; font-weight:700;" if v >= 0 else "color: red; font-weight:700;"

# ================= 요약표 헤더 + 체크박스 =================
left_col, right_col = st.columns([0.85, 0.15])
with left_col:
    st.markdown("#### 요약표")
with right_col:
    show_detail = st.checkbox("자세히 보기", value=False)

st.caption(
    f"선택기간: {start_date.date()} ~ {end_date.date()}  ·  이전기간: {prev_start.date()} ~ {prev_end.date()}  "
    f"(이전기간 라벨: {label_for_span(prev_start, prev_end)} / 선택기간 라벨: {label_for_span(start_date, end_date)})"
)

# ================= 요약표 본표 =================
styler = (
    summary_df.style
        .format(na_rep="-")
        .format("{:,.1f}", subset=idx["매출액(백만원)", ["이전기간 합계", "선택기간 합계"]])
        .format("{:,.0f}", subset=idx["승객수(명)", ["이전기간 합계", "선택기간 합계"]])
        .format("{:,.1f}%", subset=idx[:, ["증감율(%)"]])
        .applymap(style_delta, subset=idx[:, ["증감율(%)"]])
)
st.dataframe(styler, use_container_width=True)

# ================= 자세히 보기: 일일데이터(실적/예측 + 오차율) =================
if show_detail:
    # 선택기간 달력(모든 일자)
    dates_df = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    if dates_df.empty:
        st.info("선택기간 데이터가 없습니다.")
    else:
        # 1) 예측 승객수: forecast_pass.csv
        f_pass = (
            forecast_df_all[forecast_df_all["date"].between(start_date, end_date)]
            .loc[:, ["date", "passengers"]]
            .rename(columns={"passengers": "pred_passengers"})
        )

        # 2) 예측 매출액: forecast_sales.csv / .cvs
        try:
            f_sales_raw = load_forecast_sales_df()
            f_sales = (
                f_sales_raw[f_sales_raw["date"].between(start_date, end_date)]
                .copy()
            )
            f_sales["pred_sales_million"] = (
                pd.to_numeric(f_sales["pred_sales_amount"], errors="coerce") / 1_000_000
            )
            f_sales = f_sales[["date", "pred_sales_million"]]
        except FileNotFoundError as e:
            st.error(f"{e}")
            f_sales = pd.DataFrame(columns=["date", "pred_sales_million"])

        # 3) 실적(승객/매출): train_reservations_rows.csv
        try:
            act_daily_raw = load_actual_rows_df()
            act_daily = act_daily_raw[
                act_daily_raw["date"].between(start_date, end_date)
            ].copy()
            act_daily.rename(
                columns={
                    "passengers": "act_passengers",
                    "sales_amount": "act_sales_amount",
                },
                inplace=True,
            )
            act_daily["act_sales_million"] = (
                pd.to_numeric(act_daily["act_sales_amount"], errors="coerce") / 1_000_000
            )
            act_daily = act_daily[["date", "act_passengers", "act_sales_million"]]
        except (FileNotFoundError, KeyError) as e:
            st.error(f"실적 로딩 오류: {e}")
            act_daily = pd.DataFrame(columns=["date", "act_passengers", "act_sales_million"])

        # 4) 병합 (달력 기준 좌측 조인)
        daily_merged = (
            dates_df.merge(act_daily, on="date", how="left")
                    .merge(f_pass, on="date", how="left")
                    .merge(f_sales, on="date", how="left")
                    .sort_values("date")
        )

        # 5) 오차율 계산 (실적이 0/결측이면 NaN)
        daily_merged["승객수 오차율(%)"] = (
            (daily_merged["pred_passengers"] - daily_merged["act_passengers"])
            / daily_merged["act_passengers"] * 100.0
        ).where(daily_merged["act_passengers"].fillna(0).ne(0))

        daily_merged["매출액 오차율(%)"] = (
            (daily_merged["pred_sales_million"] - daily_merged["act_sales_million"])
            / daily_merged["act_sales_million"] * 100.0
        ).where(daily_merged["act_sales_million"].fillna(0).ne(0))

        # 6) 표 생성
        detail_df = pd.DataFrame({
            "날짜": daily_merged["date"].dt.strftime("%Y-%m-%d"),
            "실적 승객수(명)": daily_merged["act_passengers"],
            "실적 매출액(백만원)": daily_merged["act_sales_million"],
            "예측 승객수(명)": daily_merged["pred_passengers"],
            "예측 매출액(백만원)": daily_merged["pred_sales_million"],
            "승객수 오차율(%)": daily_merged["승객수 오차율(%)"],
            "매출액 오차율(%)": daily_merged["매출액 오차율(%)"],
        })

        # ===== 강조 스타일 함수 정의 =====
        def highlight_error(v):
            if pd.isna(v):
                return ""
            if abs(v) <= 10:   # ±10% 이내는 파란색
                return "color: blue; font-weight: 700;"
            elif abs(v) >= 20: # ±20% 이상은 빨간색
                return "color: red; font-weight: 700;"
            else:
                return ""

        # ===== 스타일 적용 =====
        st.markdown("##### 📅 선택기간 일일 데이터 (실적 vs 예측 + 오차율)")
        st.dataframe(
            detail_df.style
                .format({
                    "실적 승객수(명)": "{:,.0f}",
                    "예측 승객수(명)": "{:,.0f}",
                    "실적 매출액(백만원)": "{:,.1f}",
                    "예측 매출액(백만원)": "{:,.1f}",
                    "승객수 오차율(%)": "{:,.1f}%",
                    "매출액 오차율(%)": "{:,.1f}%"
                })
                .applymap(highlight_error, subset=["승객수 오차율(%)", "매출액 오차율(%)"]),  # ✅ 오차율 강조 적용
            use_container_width=True,
            height=360
        )
