
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sqlite3
from datetime import datetime, date

DB_PATH = "data.db"

st.set_page_config(
    page_title="エリア別入札データ可視化",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Fonts & CSS (Noto Sans JP)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;600;700&display=swap');
html, body, [class*="css"]  {
  font-family: 'Noto Sans JP', sans-serif;
}
/* KPI cards */
.kpi {
  padding: 12px 16px;
  border-radius: 12px;
  border: 1px solid rgba(0,0,0,0.1);
  background: rgba(0,0,0,0.03);
}
.small {
  font-size: 12px;
  color: #666;
}
</style>
""", unsafe_allow_html=True)

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
        """)
        # メタ情報（ユーザーが選んだ列名などを保管、将来拡張用）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                val TEXT
            );
        """)

def table_exists(conn, table_name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(q, (table_name,)).fetchone() is not None

def num_rows() -> int:
    with get_conn() as conn:
        if not table_exists(conn, "records"):
            return 0
        # recordsテーブルが空でも列が沢山ある可能性に注意
        try:
            return conn.execute("SELECT COUNT(1) FROM records").fetchone()[0]
        except:
            return 0

def infer_and_create_records_table(df: pd.DataFrame):
    """
    初回取り込み時にrecordsテーブルがない場合、dfの列構成をもとに可変スキーマで生成する。
    """
    with get_conn() as conn:
        # recordsテーブルが空なら、一旦削除→作り直し（列を柔軟に反映）
        conn.execute("DROP TABLE IF EXISTS records;")
        # 列名→SQL列定義（全てTEXTで取り込み、後段で型変換）
        cols_sql = ",\n".join([f'"{c}" TEXT' for c in df.columns])
        conn.execute(f"""
            CREATE TABLE records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {cols_sql}
            );
        """)

def append_df(df: pd.DataFrame):
    """
    recordsへ追記。列が不足していれば拡張、余計な列は無視しない（拡張して取り込む）
    """
    if df.empty:
        return 0
    with get_conn() as conn:
        # 既存テーブルの列一覧を取得
        cur_cols = []
        try:
            cur = conn.execute("PRAGMA table_info(records);").fetchall()
            cur_cols = [c[1] for c in cur]  # [cid, name, type, notnull, dflt, pk]
        except:
            pass

        # 最初の取り込み or テーブルが未定義の時は作り直し
        if not cur_cols or len(cur_cols) <= 1:  # idのみ等
            infer_and_create_records_table(df)
            cur_cols = [c[1] for c in conn.execute("PRAGMA table_info(records);").fetchall()]

        # 不足している列があれば追加（TEXT型）
        add_cols = [c for c in df.columns if c not in cur_cols]
        for c in add_cols:
            try:
                conn.execute(f'ALTER TABLE records ADD COLUMN "{c}" TEXT;')
            except Exception as e:
                st.warning(f"列追加に失敗: {c} -> {e}")

        # 取り込み
        # 文字列化してからINSERT（型は後で選択可能に）
        df_to_insert = df.copy()
        for c in df_to_insert.columns:
            df_to_insert[c] = df_to_insert[c].astype(str)
        # バルクインサート
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

def get_all_records_df(limit: int|None=None) -> pd.DataFrame:
    with get_conn() as conn:
        if not table_exists(conn, "records"):
            return pd.DataFrame()
        q = "SELECT * FROM records ORDER BY id"
        if limit and limit > 0:
            q += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(q, conn)
    # id列は内部用
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
        clear = st.button("🗑️ DBリセット（全削除）", type="secondary")

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

# 取り込み済みデータの取得
raw_df = get_all_records_df()

if raw_df.empty:
    st.info("まずは左のサイドバーからExcelを取り込んでください。複数シートは自動で縦結合され、オプションでシート名→地域列を付与できます。")
    st.stop()

# -----------------------------
# 列の役割をユーザー指定（柔軟対応）
# -----------------------------
st.subheader("🔎 列の役割設定")
cols = list(raw_df.columns)
# 推定候補
date_cand = [c for c in cols if any(k in c.lower() for k in ["date", "日付", "日時", "time", "時刻", "開始", "取引"])]
region_cand = [c for c in cols if any(k in c for k in ["地域","エリア","エリア名","供給エリア","エリアコード","エリア名","area","region","地域名"])]
numeric_cand = [c for c in cols if c not in date_cand and c not in region_cand]

c1, c2, c3 = st.columns([1,1,2])
with c1:
    date_col = st.selectbox("日付列", options=cols, index=(cols.index(date_cand[0]) if date_cand else 0))
with c2:
    region_col = st.selectbox("地域列", options=cols, index=(cols.index(region_cand[0]) if region_cand else cols.index(cols[-1])))
with c3:
    metric_cols = st.multiselect("数値列（複数選択可）", options=cols, default=[c for c in numeric_cand][:3])

# 型変換
df = raw_df.copy()
# 日付
try:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
except:
    pass
# 数値列（選択列のみ）
for m in metric_cols:
    # 数値に変換（カンマ等除去）
    df[m] = pd.to_numeric(df[m].astype(str).str.replace(",", ""), errors="coerce")

# -----------------------------
# フィルタ
# -----------------------------
st.subheader("🧰 フィルタ")
# 地域
regions = sorted([x for x in df[region_col].dropna().astype(str).unique().tolist()])
sel_regions = st.multiselect("地域を選択", options=regions, default=regions)
# 期間
valid_dates = df[date_col].dropna()
if valid_dates.empty:
    st.warning("日付列に有効な値が見つかりませんでした。日付列の指定を見直してください。")
    st.stop()
min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
r = st.slider("期間を指定", min_value=min_d, max_value=max_d, value=(min_d, max_d))
freq = st.selectbox("集計粒度", options=["日次","週次","月次"], index=2)
agg_mode = st.selectbox("集計方法", options=["平均","合計","中央値"], index=0)

# フィルタ適用
mask = (df[region_col].astype(str).isin(sel_regions)) & (df[date_col].dt.date.between(r[0], r[1]))
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("条件に合致するデータがありません。フィルタを調整してください。")
    st.stop()

# -----------------------------
# 集計ユーティリティ
# -----------------------------
def resample_frame(frame: pd.DataFrame, on: str, by_region: bool, metrics: list[str], freq: str, how: str):
    tmp = frame[[on, region_col] + metrics].dropna(subset=[on]).copy()
    tmp = tmp.sort_values(on)
    # 周波数
    rule = {"日次":"D","週次":"W","月次":"MS"}[freq]
    # 集計関数
    agg_map = {"平均":"mean","合計":"sum","中央値":"median"}[how]
    # 時系列にするためset_index
    tmp = tmp.set_index(on)
    if by_region:
        grp = tmp.groupby(region_col)
        out = []
        for g, gdf in grp:
            agg = gdf.resample(rule).agg(agg_map)
            agg[region_col] = g
            out.append(agg.reset_index())
        res = pd.concat(out, ignore_index=True)
    else:
        res = tmp.resample(rule).agg(agg_map).reset_index()
    return res

# -----------------------------
# KPI
# -----------------------------
st.subheader("📈 概要KPI")
kc1, kc2, kc3, kc4 = st.columns(4)
with kc1:
    st.markdown('<div class="kpi">総件数<br><span class="small"></span><h3>{:,}</h3></div>'.format(len(fdf)), unsafe_allow_html=True)
with kc2:
    st.markdown('<div class="kpi">期間<br><h3>{} ～ {}</h3></div>'.format(r[0].strftime("%Y-%m-%d"), r[1].strftime("%Y-%m-%d")), unsafe_allow_html=True)
with kc3:
    st.markdown('<div class="kpi">地域数<br><h3>{}</h3></div>'.format(len(sel_regions)), unsafe_allow_html=True)
with kc4:
    st.markdown('<div class="kpi">選択指標<br><h3>{}</h3></div>'.format(len(metric_cols)), unsafe_allow_html=True)

# -----------------------------
# 可視化
# -----------------------------
st.subheader("📊 可視化")

# (A) 時系列（地域別）
if metric_cols:
    for m in metric_cols:
        st.markdown(f"**時系列（{m}）**")
        ts = resample_frame(fdf, on=date_col, by_region=True, metrics=[m], freq=freq, how=agg_mode)
        # Altair
        chart = alt.Chart(ts).mark_line(point=True).encode(
            x=alt.X(f"{date_col}:T", title="日時"),
            y=alt.Y(f"{m}:Q", title=m),
            color=alt.Color(f"{region_col}:N", title="地域"),
            tooltip=[date_col, region_col, m]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# (B) 地域横比較（平均/合計/中央値）
if metric_cols:
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

# (C) 分布（ヒストグラム）
if metric_cols:
    st.markdown("**分布（ヒストグラム）**")
    m = st.selectbox("ヒストグラムの対象列", options=metric_cols, index=0)
    # クリップして極端な外れ値を軽減（5~95%）
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

# (D) ピボット（地域 × 月のマトリクス）
if metric_cols:
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
    # 集計例：月次×地域×各指標（平均）
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
