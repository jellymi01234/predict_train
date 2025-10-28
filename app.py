#1차 초안

# app.py ── Streamlit (https://<YOUR-APP>.streamlit.app)
import io
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ================= 기본 설정 =================
st.set_page_config(page_title="Passengers & Sales (Dual Axis)", layout="wide")
st.title("📈 외부요인 기반 철도수요예측 시스템")
st.caption("-- GitHub 저장소에 포함된 merged.csv를 기본으로 읽습니다 --")

# ================= 사이드바: 데이터 입력/설정 =================
st.sidebar.header("⚙️ 데이터 입력")
uploaded = st.sidebar.file_uploader("CSV 업로드(선택, 업로드 시 업로드 파일을 사용)", type=["csv"])

st.sidebar.header("🧰 옵션")
use_secondary_axis = st.sidebar.checkbox("보조축 사용(권장)", value=True)
interpolate_missing = st.sidebar.checkbox("결측치 보간(선 끊김 방지)", value=False)
use_rolling = st.sidebar.checkbox("이동평균(스무딩)", value=False)
window = st.sidebar.slider("이동평균 윈도우(일)", min_value=2, max_value=14, value=3, step=1, disabled=not use_rolling)
show_markers = st.sidebar.checkbox("마커 표시", value=False)
resample_daily = st.sidebar.checkbox("일 단위 리샘플(중복/결측 날짜 정리)", value=True)

# ================= 데이터 로드 =================
@st.cache_data(show_spinner=False)
def load_df_from_repo_csv(filename: str = "merged.csv"):
    """저장소에 포함된 CSV(상대경로)를 읽습니다."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"저장소에서 '{filename}' 파일을 찾을 수 없습니다.")
    # 인코딩 추정
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_df_from_buffer(buf):
    """업로드된 파일 버퍼를 읽습니다."""
    # 인코딩 추정
    try:
        return pd.read_csv(buf)
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf, encoding="cp949")

# 실제 로드 로직
df = None
load_error = None
if uploaded is not None:
    try:
        df = load_df_from_buffer(uploaded)
        st.success("✅ 업로드한 CSV를 불러왔습니다.")
    except Exception as e:
        load_error = f"업로드 파일을 읽는 중 오류: {e}"
else:
    try:
        df = load_df_from_repo_csv("merged.csv")
        st.success("✅ 저장소의 'merged.csv'를 불러왔습니다.")
    except Exception as e:
        load_error = f"CSV를 읽는 중 오류: {e}"

if load_error:
    st.error(load_error)
    st.stop()
if df is None or df.empty:
    st.warning("CSV를 업로드하거나 저장소에 'merged.csv'를 추가해 주세요.")
    st.stop()

# ================= 컬럼 선택 UI =================
st.sidebar.header("📑 컬럼 선택")
# 기본 추정
guess_date = "date" if "date" in df.columns else df.columns[0]
guess_pax  = "passengers" if "passengers" in df.columns else (df.columns[1] if len(df.columns) > 1 else guess_date)
guess_sales= "sales_amount" if "sales_amount" in df.columns else (df.columns[2] if len(df.columns) > 2 else guess_pax)

date_col  = st.sidebar.selectbox("날짜 컬럼", options=list(df.columns), index=list(df.columns).index(guess_date) if guess_date in df.columns else 0)
pax_col   = st.sidebar.selectbox("승객 수 컬럼", options=list(df.columns), index=list(df.columns).index(guess_pax) if guess_pax in df.columns else 0)
sales_col = st.sidebar.selectbox("매출 컬럼", options=list(df.columns), index=list(df.columns).index(guess_sales) if guess_sales in df.columns else 0)

# ================= 전처리 =================
df = df.copy()
# 날짜 변환
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col)

# 숫자 변환
for c in [pax_col, sales_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 리샘플(선택): 동일 날짜 여러 행/누락 날짜 보정
if resample_daily:
    daily = (df.set_index(date_col)[[pax_col, sales_col]]
               .resample("D")
               .sum(min_count=1))
    df_rs = daily.reset_index()
else:
    df_rs = df[[date_col, pax_col, sales_col]].copy()

# 결측 보간(선택)
if interpolate_missing:
    df_rs[[pax_col, sales_col]] = df_rs[[pax_col, sales_col]].interpolate(method="time", limit_direction="both")

# ================= 날짜 범위 UI =================
st.sidebar.header("📅 날짜 구간")
data_min = pd.to_datetime(df_rs[date_col].min()).date()
data_max = pd.to_datetime(df_rs[date_col].max()).date()
default_start = date(2024, 8, 1) if data_min <= date(2024, 8, 1) <= data_max else data_min
default_end   = date(2024, 8, 28) if data_min <= date(2024, 8, 28) <= data_max else data_max

start_end = st.sidebar.date_input(
    "시작일 / 종료일",
    value=(default_start, default_end),
    min_value=data_min, max_value=data_max
)
if isinstance(start_end, tuple):
    start_date, end_date = start_end
else:
    start_date = start_end
    end_date = start_end

mask = (df_rs[date_col] >= pd.to_datetime(start_date)) & (df_rs[date_col] <= pd.to_datetime(end_date))
plot_df = df_rs.loc[mask, [date_col, pax_col, sales_col]].copy()

if use_rolling and not plot_df.empty:
    plot_df[pax_col] = plot_df[pax_col].rolling(window=window, min_periods=1).mean()
    plot_df[sales_col] = plot_df[sales_col].rolling(window=window, min_periods=1).mean()

# ================= 그래프 =================
if plot_df.empty:
    st.warning("선택한 구간에 데이터가 없습니다. 날짜 범위를 조정해 주세요.")
    st.stop()

st.subheader("예측그래프")

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()  # 오른쪽 y축

# 색상
color_pax   = "orange"  # 승객수
color_sales = "blue"    # 매출액

# 왼쪽 y축 → 매출(선)
line_sales, = ax1.plot(
    plot_df[date_col], plot_df[sales_col],
    color=color_sales,
    linewidth=2.2,
    marker="o" if show_markers else None,
    label=str(sales_col),
)

# 오른쪽 y축 → 승객수(막대)
bars_pax = ax2.bar(
    plot_df[date_col], plot_df[pax_col],
    color=color_pax,
    alpha=0.5,
    width=0.7,
    label=str(pax_col)
)

# 축/라벨
ax1.set_xlabel("")
ax1.set_ylabel(str(sales_col), color=color_sales)
ax2.set_ylabel(str(pax_col), color=color_pax)
ax1.tick_params(axis="y", labelcolor=color_sales)
ax2.tick_params(axis="y", labelcolor=color_pax)

# 승객수 y축 범위 고정 (필요 시 조정)
ax2.set_ylim(0, 500000)

# 격자/날짜 포맷
ax1.grid(True, alpha=0.3)
fig.autofmt_xdate()

# 범례
lines = [line_sales, bars_pax]
labels = [l.get_label() for l in lines]
fig.legend(
    lines, labels,
    loc="upper right",
    bbox_to_anchor=(0.915, 0.94),
    frameon=True,
    facecolor="white",
    edgecolor="gray"
)

plt.subplots_adjust(top=0.93, right=0.93)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# ================= 요약표 (이전기간 = 선택기간과 동일 길이) =================
# 👉 같은 df를 그대로 사용 (별도 경로 재로딩 없음)

def compute_previous_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    """
    선택기간 일수 n = (end_dt - start_dt) + 1  [양끝 포함]
    이전기간 길이 = n (선택기간과 동일)
    이전기간 종료 = start_dt - 1일
    이전기간 시작 = 이전기간 종료 - (n - 1)일
    """
    n_days = (end_dt - start_dt).days + 1
    prev_len = n_days
    prev_end = start_dt - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=prev_len - 1)
    return n_days, prev_len, prev_start.normalize(), prev_end.normalize()

# 날짜 파싱 보조
def parse_dates_safe(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if out.isna().all():
        out = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return out

# 이전기간 범위
start_dt = pd.to_datetime(start_date)
end_dt   = pd.to_datetime(end_date)
n_days, prev_len, prev_start_dt, prev_end_dt = compute_previous_range(start_dt, end_dt)

# 원본 df에서 이전기간 필터 (이미 날짜/숫자 전처리됨)
hist_df = df[[date_col, sales_col, pax_col]].copy()
hist_df[date_col] = parse_dates_safe(hist_df[date_col])
hist_df = hist_df.dropna(subset=[date_col]).sort_values(date_col)

mask_prev = (hist_df[date_col] >= prev_start_dt) & (hist_df[date_col] <= prev_end_dt)
prev_df = hist_df.loc[mask_prev].sort_values(date_col)

# 합계 계산
prev_sales_total = float(pd.to_numeric(prev_df[sales_col], errors="coerce").fillna(0).sum()) if not prev_df.empty else 0.0
prev_pax_total   = float(pd.to_numeric(prev_df[pax_col],   errors="coerce").fillna(0).sum()) if not prev_df.empty else 0.0

forecast_sales_total = float(pd.to_numeric(plot_df[sales_col], errors="coerce").fillna(0).sum())
forecast_pax_total   = float(pd.to_numeric(plot_df[pax_col],   errors="coerce").fillna(0).sum())

def pct_num(forecast, previous):
    if previous == 0:
        return None
    return (forecast - previous) / previous * 100.0

sales_delta_pct = pct_num(forecast_sales_total, prev_sales_total)
pax_delta_pct   = pct_num(forecast_pax_total,   prev_pax_total)

# ================= 요약표(Styler 시각화) =================
ui_df = pd.DataFrame({
    "지표": ["매출액", "승객수"],
    "이전실적 합계": [prev_sales_total, prev_pax_total],
    "예측 합계":     [forecast_sales_total, forecast_pax_total],
    "증감률(%)":     [
        np.nan if sales_delta_pct is None else sales_delta_pct,
        np.nan if pax_delta_pct   is None else pax_delta_pct
    ],
    "기간(일수)":    [n_days, n_days],
})

def style_delta(v):
    if pd.isna(v): return ""
    color = "green" if v >= 0 else "red"
    return f"color:{color}; font-weight:700;"

styled = (
    ui_df.style
        .format({
            "이전실적 합계": "{:,.0f}",
            "예측 합계":     "{:,.0f}",
            "증감률(%)":     "{:,.1f}%",
            "기간(일수)":    "{:,.0f}",
        })
        .bar(subset=["이전실적 합계"], color="#e9ecef")
        .bar(subset=["예측 합계"], color="#cfe2ff")
        .applymap(style_delta, subset=["증감률(%)"])
)

left_col, right_col = st.columns([0.85, 0.2])
with left_col:
    st.markdown("#### 요약표")
with right_col:
    show_prev_detail = st.checkbox("자세히 보기", value=False, key="chk_prev_detail")

st.dataframe(styled, use_container_width=True)

if show_prev_detail:
    if prev_df.empty:
        st.info("이전기간 데이터가 없습니다.")
    else:
        debug_df = prev_df[[date_col, sales_col, pax_col]].copy()
        debug_df.columns = ["날짜(실적)", "매출(실적)", "승객(실적)"]
        st.dataframe(debug_df, use_container_width=True)

# ================= 다운로드 =================
st.subheader("⬇️ 내보내기")

# PNG 저장 & 다운로드
buf_png = io.BytesIO()
fig.savefig(buf_png, format="png", dpi=160, bbox_inches="tight")
buf_png.seek(0)
st.download_button(
    label="그래프 PNG 다운로드",
    data=buf_png,
    file_name=f"dual_axis_{start_date}_{end_date}.png",
    mime="image/png",
)

# CSV(필터/전처리 결과) 다운로드
csv_bytes = plot_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="필터 데이터 CSV 다운로드",
    data=csv_bytes,
    file_name=f"filtered_{start_date}_{end_date}.csv",
    mime="text/csv",
)

# 로컬 저장(선택) ─ 상대경로로 변경
with st.expander("💾 로컬에 저장(선택)", expanded=False):
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"dual_axis_{start_date}_{end_date}.png"
    out_csv = out_dir / f"filtered_{start_date}_{end_date}.csv"
    save_local = st.toggle("로컬 저장 실행", value=False)
    if save_local:
        with open(out_png, "wb") as f:
            f.write(buf_png.getvalue())
        plot_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        st.success(f"저장 완료:\n- {out_png}\n- {out_csv}")

st.info("팁: 스케일 차이가 크면 보간/이동평균을 켜서 추세를 보기 좋게 만들 수 있어요.")
