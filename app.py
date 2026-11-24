# -*- coding: utf-8 -*-
"""
GreenPower Lite —— 绿电直连仿真轻量版（云端部署包）
仅包含 固定策略 / LP松弛 两个计算模块，无技经 Excel、无寻优、无 MILP。

技术口径与主程序完全一致：
  - 8760小时4列数据文件（单位光伏出力/单位风电出力/用户负荷/绿电结算电价）
  - 发电量 = 有效装机(kW) × 单位出力曲线
  - 储能充放时长决定功率上限，SOC 递推含充放电效率
  - 专线线损作用于全部送端电量（受端口径计量）
  - 求解器默认 CBC（PuLP 内置，与主应用一致），可选 HiGHS
"""
import streamlit as st
import numpy as np
import pandas as pd

from fixed_strategy import fixed_strategy_simulation
from lp_strategy import lp_strategy_simulation

# ===== 页面配置：国家电投品牌色（中国红 + 梦想绿，白底）=====
st.set_page_config(
    page_title="GreenPower Lite - 绿电直连仿真",
    page_icon="⚡",
    layout="wide",
)
st.markdown(
    """
    <style>
    h1 { color: #C7000B; }            /* 中国红 */
    [data-testid="stMetricValue"] { color: #00843D; }  /* 梦想绿 */
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ GreenPower Lite · 绿电直连仿真")
st.caption("固定策略 / LP调度 · 全年8760小时仿真 · 消纳与收入测算")

# ===================== 侧栏：数据与参数 =====================
with st.sidebar:
    st.header("📁 8760时序数据")
    st.markdown(
        "上传 Excel(.xlsx/.xls) 或 CSV，需 **8760行4列**：\n\n"
        "单位光伏出力(kW/kW)、单位风电出力(kW/kW)、"
        "用户负荷(kW)、绿电结算电价(元/kWh)"
    )
    uploaded = st.file_uploader("选择文件", type=["xlsx", "xls", "csv"])

    st.header("⚙️ 装机参数")
    pv_mw = st.number_input("光伏装机(MW)", 0.0, 10000.0, 0.0, 1.0)
    wt_mw = st.number_input("风电装机(MW)", 0.0, 10000.0, 0.0, 1.0)
    es_mwh = st.number_input("储能容量(MWh)", 0.0, 10000.0, 0.0, 1.0)
    es_hours = st.number_input("储能充放时长(h)", 0.5, 24.0, 2.0, 0.5)
    es_ce = st.number_input("充电效率", 0.0, 1.0, 0.95, 0.01)
    es_de = st.number_input("放电效率", 0.0, 1.0, 0.95, 0.01)
    soc_min = st.number_input("SOC下限", 0.0, 1.0, 0.10, 0.01)
    soc_max = st.number_input("SOC上限", 0.0, 1.0, 1.0, 0.01)
    soc0 = st.number_input("初始SOC", 0.0, 1.0, 0.50, 0.01)
    line_loss = st.number_input("专线线损率(%)", 0.0, 50.0, 5.0, 0.1) / 100.0

    st.header("🧮 计算设置")
    strategy = st.radio("调度策略", ["固定策略", "LP松弛策略"],
                        help="固定策略：盈余充电、缺额放电；LP：全年总收入最大化调度")
    solver_name = st.selectbox("LP求解器", ["cbc", "highs"], index=0,
                               help="CBC 为 PuLP 内置默认；HiGHS 需安装 highspy")
    solver_timeout = st.number_input("求解超时(s)", 10, 3600, 300, 10)

# ===================== 数据读取与校验 =====================
curves = None
if uploaded is not None:
    try:
        fn = uploaded.name.lower()
        if fn.endswith((".xlsx", ".xls")):
            df = pd.read_excel(
                uploaded,
                engine="openpyxl" if fn.endswith(".xlsx") else "xlrd")
        else:
            df = None
            for enc in ["utf-8", "gbk", "gb2312", "gb18030", "latin1"]:
                try:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if df is None:
                st.sidebar.error("编码无法识别，请另存为 UTF-8")
        if df is not None:
            df.columns = df.columns.str.strip()
            req = ["单位光伏出力(kW/kW)", "单位风电出力(kW/kW)",
                   "用户负荷(kW)", "绿电结算电价(元/kWh)"]
            missing = [c for c in req if c not in df.columns]
            if len(df) != 8760:
                st.sidebar.error(f"行数错误：当前{len(df)}行，必须8760行")
            elif missing:
                st.sidebar.error(f"缺少列：{missing}")
            elif df[req].isna().any().any() or (df[req] < 0).any().any():
                st.sidebar.error("存在缺失值或负数")
            else:
                curves = {c: df[c].to_numpy(dtype=np.float64)
                          for c in req}
                st.sidebar.success("数据校验通过")
    except Exception as e:
        st.sidebar.error(f"读取失败：{type(e).__name__}: {e}")

# ===================== 主区：运行与结果 =====================
if curves is None:
    st.info("请在左侧上传 8760 小时时序数据文件后开始计算")
    st.stop()

can_run = (pv_mw + wt_mw) > 0
run = st.button("🚀 开始计算", type="primary", disabled=not can_run)
if not can_run:
    st.warning("光伏/风电装机至少填写一项")

if run:
    # MW → kW（与主程序换算口径一致）
    spec = dict(
        pv_installed=pv_mw * 1000.0, wt_installed=wt_mw * 1000.0,
        es_capacity=es_mwh * 1000.0, es_charge_hours=es_hours,
        es_charge_eff=es_ce, es_discharge_eff=es_de,
        soc_min=soc_min, soc_max=soc_max, soc0=soc0,
        line_loss_rate=line_loss,
    )
    kwargs = dict(
        pv_unit_power=curves["单位光伏出力(kW/kW)"],
        wt_unit_power=curves["单位风电出力(kW/kW)"],
        load_profile=curves["用户负荷(kW)"],
        net_price_profile=curves["绿电结算电价(元/kWh)"],
        **{k: spec[k] for k in
           ["pv_installed", "wt_installed", "es_capacity",
            "es_charge_hours", "es_charge_eff", "es_discharge_eff",
            "soc_min", "soc_max", "line_loss_rate"]},
        soc_initial=spec["soc0"],
    )
    with st.spinner("计算中（LP 约需数秒~数十秒）..."):
        if strategy == "固定策略":
            rdf, summ = fixed_strategy_simulation(**kwargs)
        else:
            rdf, summ = lp_strategy_simulation(
                **kwargs, solver_timeout=solver_timeout,
                solver_name=solver_name, solver_msg=0)

    # ===== 核心指标 =====
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("全年总收入(万元)", f"{summ['全年总收入(万元)']:.1f}")
    c2.metric("总售电均价(元/kWh)", f"{summ['总售电均价(元/kWh)']:.4f}")
    c3.metric("绿电消纳量(MWh)", f"{summ['全年总绿电消纳量(MWh)']:.0f}")
    c4.metric("绿电消纳率", f"{summ['绿电消纳率'] * 100:.1f}%")
    c5.metric("负荷满足率", f"{summ['负荷满足率'] * 100:.1f}%")
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("总发电量(MWh)", f"{summ['全年总发电量(MWh)']:.0f}")
    c7.metric("总弃电量(MWh)", f"{summ['全年总弃电量(MWh)']:.0f}")
    c8.metric("专线损耗(MWh)", f"{summ['全年专线总损耗(MWh)']:.0f}")
    c9.metric("缺口电量(MWh)", f"{summ['全年总缺口电量(MWh)']:.0f}")
    c10.metric("策略", summ["策略类型"].replace("策略", ""))

    # ===== 汇总明细 =====
    st.subheader("📋 分项汇总")
    left, right = st.columns(2)
    with left:
        st.markdown("**光伏 / 风电分项 (MWh)**")
        st.dataframe(pd.DataFrame(
            [summ["光伏分项"], summ["风电分项"]],
            index=["光伏", "风电"]).T, use_container_width=True)
    with right:
        st.markdown("**储能分项 (MWh)**")
        st.dataframe(pd.DataFrame([summ["储能分项"]]).T,
                     use_container_width=True)

    # ===== 曲线对比图：纵坐标固定不缩放，横坐标显示日期，多曲线叠加 =====
    st.subheader("📈 逐时曲线（默认展示前7天）")
    n_days = st.slider("展示天数", 1, 30, 7)
    view = rdf.head(n_days * 24).set_index("时间")
    chart_df = view[["风光总发电量(kW)", "用户负荷(kW)",
                     "绿电消纳量(kWh)"]]
    st.line_chart(chart_df, height=320)
    if es_mwh > 0:
        st.caption("储能充放功率（充电为负、放电为正）")
        st.line_chart(view["储能功率(kW)"], height=240)

    # ===== 下载 =====
    st.subheader("💾 结果下载")
    csv_hour = rdf.to_csv(index=False, encoding="utf-8-sig").encode()
    st.download_button("下载逐时明细表 (CSV)", csv_hour,
                       "逐时明细.csv", "text/csv")
    summ_flat = pd.Series(
        {k: v for k, v in summ.items()
         if not isinstance(v, dict)}).to_frame("数值")
    csv_summ = summ_flat.to_csv(encoding="utf-8-sig").encode()
    st.download_button("下载汇总表 (CSV)", csv_summ,
                       "汇总统计.csv", "text/csv")

st.divider()
st.caption("GreenPower Lite · 固定策略/LP仿真 · 与主程序计算口径一致 · CBC/HiGHS 求解")
