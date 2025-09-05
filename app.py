
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sqlite3
from datetime import datetime, date, time
from typing import Optional

DB_PATH = "data.db"

st.set_page_config(
    page_title="エリア別入札データ可視化",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 エリア別入札データ 可視化ダッシュボード")

# -----------------------------
# DB helpers
# -----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            );
        """ )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                val TEXT
            );
        """ )

def table_exists(conn, table_name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(q, (table_name,)).fetchone() is not None

def num_rows() -> int:
    with get_conn() as conn:
        if not table_exists(conn, "records"):
            return 0
        try:
            return conn.execute("SELECT COUNT(1) FROM records").fetchone()[0]
        except:
            return 0

def infer_and_create_records_table(df: pd.DataFrame):
    with get_conn() as conn:
        conn.execute("DROP TABLE IF EXISTS records;")
        cols_sql = ",\n".join([f'"{c}" TEXT' for c in df.columns])
        conn.execute(f"""
            CREATE TABLE records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {cols_sql}
            );
        """ )

def append_df(df: pd.DataFrame):
    if df.empty:
        return 0
    with get_conn() as conn:
        cur_cols = []
        try:
            cur = conn.execute("PRAGMA table_info(records);").fetchall()
            cur_cols = [c[1] for c in cur]
        except:
            pass

        if not cur_cols or len(cur_cols) <= 1:
            infer_and_create_records_table(df)
            cur_cols = [c[1] for c in conn.execute("PRAGMA table_info(records);").fetchall()]

        add_cols = [c for c in df.columns if c not in cur_cols]
        for c in add_cols:
            try:
                conn.execute(f'ALTER TABLE records ADD COLUMN "{c}" TEXT;')
            except Exception as e:
                st.warning(f"列追加に失敗: {c} -> {e}")

        df_to_insert = df.copy()
        for c in df_to_insert.columns:
            df_to_insert[c] = df_to_insert[c].astype(str)

        placeholders = ",".join(["?"] * len(df_to_insert.columns))
        colnames = ",".join([f'"{c}"' for c in df_to_insert.columns])
        conn.executemany(
            f'INSERT INTO records ({colnames}) VALUES ({placeholders})',
            df_to_insert.values.tolist()
        )
        conn.commit()
        return len(df_to_insert)

def load_excel_all_sheets(file, add_region_from_sheet=True, region_col_name="地域"):
    xls = pd.ExcelFile(file)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if add_region_from_sheet:
            df[region_col_name] = str(sheet)
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def get_all_records_df(limit: Optional[int]=None) -> pd.DataFrame:
    with get_conn() as conn:
        if not table_exists(conn, "records"):
            return pd.DataFrame()
        q = "SELECT * FROM records ORDER BY id"
        if limit and limit > 0:
            q += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(q, conn)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df

def reset_db():
    with get_conn() as conn:
        conn.execute("DROP TABLE IF EXISTS records;")
        conn.execute("DROP TABLE IF EXISTS meta;")
    init_db()

# -----------------------------
# Init
# -----------------------------
init_db()

with st.sidebar:
    st.subheader("⚙️ データ管理")
    st.caption("Excel（複数シート可）を取り込み、内部DB(SQLite)に保存します。以後はDBから高速に可視化できます。")
    uploaded = st.file_uploader("Excelファイルを選択（.xlsx）", type=["xlsx"], accept_multiple_files=True)
    add_region = st.checkbox("シート名を地域名として付与する", value=True)
    region_col_name = st.text_input("地域列の列名", value="地域")
    col1, col2 = st.columns(2)
    with col1:
        go = st.button("📥 取り込み/追加")
    with col2:
        # Remove type='secondary' for compatibility
        clear = st.button("🗑️ DBリセット（全削除）")

    if clear:
        reset_db()
        st.success("DBを初期化しました。")

    if go and uploaded:
        total = 0
        for up in uploaded:
            try:
                df = load_excel_all_sheets(up, add_region_from_sheet=add_region, region_col_name=region_col_name)
                total += append_df(df)
            except Exception as e:
                st.error(f"{up.name} の取り込みに失敗: {e}")
        st.success(f"取り込み完了: {total} 行を追加しました。")

    st.write("---")
    st.caption("📦 現在のDB件数")
    st.metric(label="総レコード数", value=f"{num_rows():,}")

raw_df = get_all_records_df()
if raw_df.empty:
    st.info("まずは左のサイドバーからExcelを取り込んでください。")
    st.stop()

# -----------------------------
# 列の役割設定
# -----------------------------
st.subheader("🔎 列の役割設定")

cols = list(raw_df.columns)

# 優先: 'ym' を日付候補に含める（先頭に置く）
date_cand = []
for c in cols:
    if c.lower() == "ym":
        date_cand.append(c)
        break
date_cand += [c for c in cols if any(k in c.lower() for k in ["date", "日付", "日時", "time", "時刻", "開始", "取引"]) and c not in date_cand]

region_cand = [c for c in cols if any(k in c for k in ["地域","エリア","エリア名","供給エリア","エリアコード","area","region","地域名"])]
numeric_cand = [c for c in cols if c not in date_cand and c not in region_cand]

def _idx_or_default(opts, target):
    try:
        return opts.index(target)
    except:
        return 0

default_date_col = date_cand[0] if date_cand else cols[0]
default_region_col = region_cand[0] if region_cand else cols[-1]

c1, c2, c3 = st.columns([1,1,2])
with c1:
    date_col = st.selectbox("日付列", options=cols, index=_idx_or_default(cols, default_date_col))
with c2:
    region_col = st.selectbox("地域列", options=cols, index=_idx_or_default(cols, default_region_col))
with c3:
    metric_cols = st.multiselect("数値列（複数選択可）", options=cols, default=[c for c in numeric_cand][:3])

# 追加機能：ym列の3時間刻み再構築
st.markdown("**⏱️ ym列の時間再構築（3時間刻み）**")
rebuild = st.checkbox("先頭を 2024-04-01 00:00、以降180分ずつインクリメントで再構築（地域ごと）", value=("ym" in [c.lower() for c in cols]))
start_date = st.date_input("開始日", value=date(2024,4,1))
start_time = st.time_input("開始時刻", value=time(0,0))

df = raw_df.copy()

if rebuild:
    start_dt = datetime.combine(start_date, start_time)
    try:
        df["_row_order_"] = np.arange(len(df))
        df[region_col] = df[region_col].astype(str)
        out = []
        for g, gdf in df.groupby(region_col, sort=False):
            gdf = gdf.sort_values("_row_order_").copy()
            rng = pd.date_range(start=start_dt, periods=len(gdf), freq="180min")
            gdf[date_col] = pd.to_datetime(rng)
            out.append(gdf)
        df = pd.concat(out, ignore_index=True).sort_values("_row_order_").drop(columns=["_row_order_"])
    except Exception as e:
        st.warning(f"ym再構築でエラー: {e}")

try:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
except Exception as e:
    st.warning(f"日付変換でエラー: {e}")
for m in metric_cols:
    df[m] = pd.to_numeric(df[m].astype(str).str.replace(",", ""), errors="coerce")

# -----------------------------
# フィルタ
# -----------------------------
st.subheader("🧰 フィルタ")
regions = sorted([x for x in df[region_col].dropna().astype(str).unique().tolist()])
sel_regions = st.multiselect("地域を選択", options=regions, default=regions)

valid_dates = df[date_col].dropna()
if valid_dates.empty:
    st.warning("日付列に有効な値が見つかりませんでした。日付列の指定や再構築設定を見直してください。")
    st.stop()
min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
if min_d > max_d:
    min_d, max_d = max_d, min_d
r = st.slider("期間を指定", min_value=min_d, max_value=max_d, value=(min_d, max_d))
freq = st.selectbox("集計粒度", options=["生データ(180分)","日次","週次","月次"], index=1)
agg_mode = st.selectbox("集計方法", options=["平均","合計","中央値"], index=0)

mask = (df[region_col].astype(str).isin(sel_regions)) & (df[date_col].dt.date.between(r[0], r[1]))
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("条件に合致するデータがありません。フィルタを調整してください。")
    st.stop()

# -----------------------------
# 集計ユーティリティ
# -----------------------------
def resample_frame(frame: pd.DataFrame, on: str, by_region: bool, metrics: list, freq: str, how: str):
    """
    Resample with selectable frequency. For '生データ(180分)', bypass aggregation and return raw 180min data.
    """
    tmp = frame[[on, region_col] + metrics].dropna(subset=[on]).copy()
    tmp = tmp.sort_values(on)
    # Map label -> pandas rule; RAW means no resample
    rule_map = {"生データ(180分)": "RAW", "日次": "D", "週次": "W", "月次": "MS"}
    rule = rule_map.get(freq, "MS")
    agg_map = {"平均":"mean","合計":"sum","中央値":"median"}[how]

    # Ensure datetime index
    tmp = tmp.set_index(on)

    # Cast metric columns to numeric again (safety)
    for m in metrics:
        tmp[m] = pd.to_numeric(tmp[m], errors="coerce")

    if rule == "RAW":
        # Return raw rows (no resample). Keep needed columns.
        if by_region:
            res = tmp.reset_index()[[on, region_col] + metrics]
        else:
            res = tmp.reset_index()[[on] + metrics]
        return res

    if by_region:
        out = []
        for g, gdf in tmp.groupby(region_col):
            num = gdf[metrics].resample(rule).agg(agg_map)
            num[region_col] = g
            num = num.reset_index()
            out.append(num)
        res = pd.concat(out, ignore_index=True)
    else:
        res = tmp[metrics].resample(rule).agg(agg_map).reset_index()
    return res

# -----------------------------
# KPI
# -----------------------------
st.subheader("📈 概要KPI")
kc1, kc2, kc3, kc4 = st.columns(4)
with kc1:
    st.metric("総件数", f"{len(fdf):,}")
with kc2:
    st.metric("期間", f"{r[0].strftime('%Y-%m-%d')} ～ {r[1].strftime('%Y-%m-%d')}")
with kc3:
    st.metric("地域数", f"{len(sel_regions)}")
with kc4:
    st.metric("選択指標数", f"{len(metric_cols)}")

# -----------------------------
# 可視化
# -----------------------------
st.subheader("📊 可視化")

if metric_cols:
    for m in metric_cols:
        st.markdown(f"**時系列（{m}）**")
        ts = resample_frame(fdf, on=date_col, by_region=True, metrics=[m], freq=freq, how=agg_mode)
        chart = alt.Chart(ts).mark_line(point=True).encode(
            x=alt.X(f"{date_col}:T", title="日時"),
            y=alt.Y(f"{m}:Q", title=m),
            color=alt.Color(f"{region_col}:N", title="地域"),
            tooltip=[date_col, region_col, m]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("**地域比較（期間内の集計値）**")
    agg_func = {"平均":"mean","合計":"sum","中央値":"median"}[agg_mode]
    comp = fdf.groupby(region_col)[metric_cols].agg(agg_func).reset_index()
    chart = alt.Chart(comp.melt(id_vars=[region_col], var_name="項目", value_name="値")).mark_bar().encode(
        x=alt.X("項目:N", title="項目"),
        y=alt.Y("値:Q", title="値"),
        color=alt.Color(f"{region_col}:N", title="地域"),
        column=alt.Column(f"{region_col}:N", title="地域")
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

    st.markdown("**分布（ヒストグラム）**")
    m = st.selectbox("ヒストグラムの対象列", options=metric_cols, index=0)
    series = fdf[m].dropna()
    if len(series) > 0:
        q5, q95 = np.nanpercentile(series, 5), np.nanpercentile(series, 95)
        hist_df = pd.DataFrame({m: series.clip(q5, q95)})
        chart = alt.Chart(hist_df).mark_bar().encode(
            x=alt.X(f"{m}:Q", bin=alt.Bin(maxbins=40), title=m),
            y=alt.Y("count():Q", title="件数")
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("ヒストグラム対象の有効データがありません。")

    st.markdown("**ピボット（地域 × 月）**")
    m = st.selectbox("ピボット表示の対象列", options=metric_cols, index=0, key="pivot_metric")
    tmp = fdf[[date_col, region_col, m]].dropna(subset=[date_col]).copy()
    tmp["月"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
    import numpy as _np
    agg = tmp.pivot_table(index=region_col, columns="月", values=m, aggfunc=_np.mean)
    st.dataframe(agg.style.format("{:,.2f}"), use_container_width=True)

# -----------------------------
# データ出力
# -----------------------------
st.subheader("⬇️ データ出力")
colx, coly = st.columns(2)
with colx:
    st.download_button(
        "フィルタ後データ（CSV）をダウンロード",
        data=fdf.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered.csv",
        mime="text/csv"
    )
with coly:
    if metric_cols:
        t2 = fdf.copy()
        t2["月"] = t2[date_col].dt.to_period("M").dt.to_timestamp()
        out = t2.groupby(["月", region_col])[metric_cols].mean().reset_index().sort_values(["月", region_col])
        st.download_button(
            "月次集計（平均, CSV）をダウンロード",
            data=out.to_csv(index=False).encode("utf-8-sig"),
            file_name="monthly_summary_mean.csv",
            mime="text/csv"
        )

st.caption("© Streamlit app template for area bids by region (JP).")

# ---- Numeric-only hardening for metrics ----
def _coerce_numeric_columns(df_in: pd.DataFrame, cols: list[str]):
    safe_cols = []
    dropped_cols = []
    for c in cols:
        ser = pd.to_numeric(df_in[c].astype(str).str.replace(",", ""), errors="coerce")
        if ser.notna().any():
            df_in[c] = ser
            safe_cols.append(c)
        else:
            dropped_cols.append(c)
    return df_in, safe_cols, dropped_cols

