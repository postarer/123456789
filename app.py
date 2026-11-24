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
  - 运行曲线使用主程序 visualization.plot_multi_results 原样渲染
展示层工具函数逐字复制自主程序（core/display_utils.py、core/ui_blocks.py、
utils.py、file_handler.py），保证界面与结果展示和主程序单方案模式同构。
"""
import time

import numpy as np
import pandas as pd
import pulp
import streamlit as st

from fixed_strategy import fixed_strategy_simulation
from lp_strategy import lp_strategy_simulation
from visualization import plot_multi_results

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# ===================== 展示层工具（逐字复制自 core/display_utils.py） =====================
ES_DISCOUNT_DEFAULT = 1.00


def es_discount():
    """读取侧边栏折扣系数（0~1），默认 1.00（不打折）"""
    try:
        v = float(st.session_state.get("es_rev_discount", ES_DISCOUNT_DEFAULT))
    except Exception:
        v = ES_DISCOUNT_DEFAULT
    return min(max(v, 0.0), 1.0)


def discounted_income(d):
    """折扣后总收入 = 风光直供收入 + 储能放电收入 × 折扣。
    兼容 dict（策略 summary）与下划线命名（批量结果行）；
    旧结果缺少收入分解字段时回退为原始全年总收入。"""
    total = d.get("全年总收入(万元)")
    if total is None:
        total = d.get("全年总收入_万元")
    es_rev = d.get("储能放电收入(万元)")
    if es_rev is None:
        es_rev = d.get("储能放电收入_万元")
    pv_wt_rev = d.get("风光直供收入(万元)")
    if pv_wt_rev is None:
        pv_wt_rev = d.get("风光直供收入_万元")
    if es_rev is None or pv_wt_rev is None:
        return float(total) if total is not None else None
    if pd.isna(es_rev) or pd.isna(pv_wt_rev):
        return float(total) if total is not None and not pd.isna(total) else None
    return float(pv_wt_rev) + float(es_rev) * es_discount()


def apply_es_discharge_discount(df):
    """逐时展示层折扣（原列一律不动，只新增列）。
    折扣口径 = 用户消纳电量（放电量×(1−线损)），与储能放电收入结算口径一致：
      - 储能消纳量(折后)(kWh) = 储能消纳量(kWh) × 折扣；
      - 当期收入(折后)(元) = 当期收入 − 储能消纳量×(1−折扣)×净电价
        （风光直供收入不变，仅储能电量按折后计价；随 CSV/Excel 下载一并导出）。
    仿真计算本身不受影响；技经是否用折后口径由高级设置开关决定。"""
    d = es_discount()
    if "储能消纳量(kWh)" in df.columns:
        df["储能消纳量(折后)(kWh)"] = df["储能消纳量(kWh)"].astype(float) * d
        if d < 1.0 and "当期收入(元)" in df.columns and "净电价(元/kWh)" in df.columns:
            df["当期收入(折后)(元)"] = (
                df["当期收入(元)"].astype(float)
                - df["储能消纳量(kWh)"].astype(float) * (1.0 - d)
                * df["净电价(元/kWh)"].astype(float))
    return df


def es_detail_with_discount(es_detail):
    """储能分项表：在 用户消纳电量(MWh) 行后插入折后行（仅展示）。
    折扣口径为用户消纳电量（放电量×(1−线损)）；旧结果缺少该键时回退总放电量。"""
    df = pd.DataFrame.from_dict(es_detail, orient="index", columns=["数值"])
    df["数值"] = pd.to_numeric(df["数值"], errors="coerce").round(4)
    if "用户消纳电量(MWh)" in df.index:
        anchor, label = "用户消纳电量(MWh)", "用户消纳电量"
    elif "总放电量(MWh)" in df.index:
        anchor, label = "总放电量(MWh)", "总放电量"
    else:
        return df
    d = es_discount()
    row = pd.DataFrame(
        {"数值": [round(float(df.loc[anchor, "数值"]) * d, 4)]},
        index=[f"{label}(折后×{d:.2f})(MWh)"])
    i = df.index.get_loc(anchor) + 1
    df = pd.concat([df.iloc[:i], row, df.iloc[i:]])
    return df


# ===================== 时间范围选择器（逐字复制自 core/ui_blocks.py） =====================
def pick_time_range(key_prefix=None):
    """曲线时间范围选择器：全年 / 指定月份 / 指定日期范围，返回 (start, end) 字符串。

    key_prefix：同页可能存在多个实例时传入控件 key 前缀（如批量按方案号），
    传 None 则使用 Streamlit 自动 key（单实例场景）。
    """
    def _k(name):
        return f"{key_prefix}{name}" if key_prefix else None

    st.subheader("时间范围选择")
    range_mode = st.radio("选择方式", ["全年", "指定月份", "指定日期范围"],
                          horizontal=True, key=_k("range_mode"))
    if range_mode == "全年":
        return "2026-01-01", "2026-12-31 23:00"
    if range_mode == "指定月份":
        month = st.selectbox("选择月份", range(1, 13),
                             format_func=lambda x: f"{x}月", key=_k("month"))
        ms = pd.Timestamp(f"2026-{month:02d}-01")
        me = (pd.Timestamp(f"2026-{month + 1:02d}-01") - pd.Timedelta(hours=1)
              if month < 12 else pd.Timestamp("2026-12-31 23:00"))
        return str(ms), str(me)
    dc1, dc2 = st.columns(2)
    with dc1:
        sd = st.date_input("开始日期", pd.Timestamp("2026-01-01"),
                           min_value=pd.Timestamp("2026-01-01"),
                           max_value=pd.Timestamp("2026-12-31"), key=_k("sd"))
    with dc2:
        ed = st.date_input("结束日期", pd.Timestamp("2026-12-31"),
                           min_value=pd.Timestamp("2026-01-01"),
                           max_value=pd.Timestamp("2026-12-31"), key=_k("ed"))
    return str(sd), str(ed) + " 23:00"


def download_file(label, path, mime=XLSX_MIME, key=None):
    """读取本地文件并渲染全宽下载按钮（file_name 取文件名）"""
    with open(path, "rb") as f:
        st.download_button(label, f, file_name=path,
                           mime=mime, key=key, use_container_width=True)


# ===================== 净电价（逐字复制自 utils.py） =====================
def preprocess_price(price_profile, time_idx, monthly_fees):
    """
    逐时刻扣减对应月份的系统运行费+政府基金附加+线损费，得到净电价
    monthly_fees: dict, key=月份1-12, value=(系统运行费, 政府基金附加, 线损费) 单位：元/kWh
    """
    months = time_idx.month.values
    net_price = np.zeros_like(price_profile, dtype=np.float64)
    for t in range(len(price_profile)):
        m = months[t]
        fee_total = sum(monthly_fees[m])
        net_price[t] = price_profile[t] - fee_total
    return net_price


# ===================== 文件上传与校验（逐字复制自 file_handler.py） =====================
def upload_and_validate_file():
    st.subheader("📁 上传8760时序数据文件")
    st.markdown("""
    支持 Excel(.xlsx/.xls) 或 CSV(.csv)，8760行，必须包含4列：
    单位光伏出力(kW/kW)、单位风电出力(kW/kW)、用户负荷(kW)、绿电结算电价(元/kWh)
    """)
    uploaded_file = st.file_uploader("选择文件", type=["xlsx", "xls", "csv"])
    if uploaded_file is None:
        st.info("请上传你的8760时序数据文件")
        return None

    try:
        fn = uploaded_file.name.lower()
        if fn.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, engine="openpyxl" if fn.endswith(".xlsx") else "xlrd")
            st.success(f"✅ Excel读取成功（{uploaded_file.name}）")
        else:
            df = None
            for enc in ["utf-8", "gbk", "gb2312", "gb18030", "latin1"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    st.success(f"✅ CSV读取成功，编码：{enc}")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if df is None:
                st.error("❌ 编码无法识别，请另存为UTF-8或改用Excel")
                return None

        if len(df) != 8760:
            st.error(f"❌ 行数错误：当前{len(df)}行，必须8760行")
            return None

        df.columns = df.columns.str.strip()
        req = ["单位光伏出力(kW/kW)", "单位风电出力(kW/kW)", "用户负荷(kW)", "绿电结算电价(元/kWh)"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            st.error(f"❌ 缺少列：{missing}")
            st.info("当前列名：" + "、".join(df.columns.tolist()))
            return None

        if df[req].isna().any().any():
            st.error("❌ 存在缺失值")
            return None
        if (df[req] < 0).any().any():
            st.error("❌ 存在负数")
            return None

        pv = df["单位光伏出力(kW/kW)"].to_numpy(dtype=np.float64).flatten()
        wt = df["单位风电出力(kW/kW)"].to_numpy(dtype=np.float64).flatten()
        ld = df["用户负荷(kW)"].to_numpy(dtype=np.float64).flatten()
        pr = df["绿电结算电价(元/kWh)"].to_numpy(dtype=np.float64).flatten()

        time_idx = pd.date_range(start="2026-01-01 00:00:00", periods=8760, freq="h")
        st.success("✅ 数据校验通过")
        with st.expander("📊 数据预览（前24小时）"):
            st.dataframe(df.head(24), use_container_width=True)
        return pv, wt, ld, pr, time_idx
    except Exception as e:
        st.error(f"❌ 读取失败：{type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ===================== 仿真模型选项（与主程序 single_mode.py 一致，Lite 仅保留两个） =====================
MODEL_OPTIONS = [
    ("固定策略", "fixed", "秒级，固定充放规则，适合快速对比"),
    ("LP松弛", "lp", "中速，线性规划寻优，无充放电互斥"),
]
MODEL_FN_MAP = {
    "fixed": fixed_strategy_simulation,
    "lp": lp_strategy_simulation,
}


# ===================== 结果渲染（与主程序 modes/single_mode.py render 同构） =====================
def render_results():
    p = st.session_state["single_params"]
    pv_inst, wt_inst, es_cap = p["pv_inst"], p["wt_inst"], p["es_cap"]
    es_ce, es_de = p["es_ce"], p["es_de"]
    soc_min, soc_max, soc0 = p["soc_min"], p["soc_max"], p["soc0"]
    line_loss_rate = p["line_loss_rate"]

    results = st.session_state["results"]
    keys = list(results.keys())

    st.markdown("---")
    st.header("📊 仿真结果汇总")

    # 1. 多方案对比总表
    st.subheader("📋 方案核心指标对比")
    compare_rows = []
    for k in keys:
        s = results[k][1]
        compare_rows.append({
            "方案": k,
            "全年总收入(万元)": round(s["全年总收入(万元)"], 2),
            "风光直供收入(万元)": round(s.get("风光直供收入(万元)", 0.0), 2),
            "储能放电收入(万元)": round(s.get("储能放电收入(万元)", 0.0), 2),
            "折扣后总收入(万元)": round(discounted_income(s), 2),
            "绿电消纳量(MWh)": round(s["全年总绿电消纳量(MWh)"], 2),
            "专线损耗(MWh)": round(s["全年专线总损耗(MWh)"], 2),
            "弃电量(MWh)": round(s["全年总弃电量(MWh)"], 2),
            "缺口电量(MWh)": round(s["全年总缺口电量(MWh)"], 2),
            "负荷满足率": f"{s['负荷满足率']:.2%}",
            "光伏直售均价": f"{s['光伏直售均价(元/kWh)']:.4f}",
            "风电直售均价": f"{s['风电直售均价(元/kWh)']:.4f}",
            "储能放电均价": f"{s['储能放电均价(元/kWh)']:.4f}",
            "总售电均价": f"{s['总售电均价(元/kWh)']:.4f}",
        })
    compare_df = pd.DataFrame(compare_rows)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # 2. 方案选择与核心指标
    st.markdown("")
    st.subheader("🔍 单方案详情")
    default_idx = 0
    if st.session_state["run_mode"] in ["multi", "milp_multi", "multi_model"]:
        max_income_key = max(keys, key=lambda k: results[k][1]["全年总收入(万元)"])
        default_idx = keys.index(max_income_key)

    selected_key = st.selectbox("选择查看方案", keys, index=default_idx)
    rdf, summ = results[selected_key]

    # 核心指标卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("全年总收入", f"{summ['全年总收入(万元)']:.2f}万元")
    c2.metric("全年专线损耗", f"{summ['全年专线总损耗(MWh)']:.2f}MWh")
    c3.metric("光伏直售均价", f"{summ['光伏直售均价(元/kWh)']:.4f}元")
    c4.metric("风电直售均价", f"{summ['风电直售均价(元/kWh)']:.4f}元")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("储能放电均价", f"{summ['储能放电均价(元/kWh)']:.4f}元")
    c6.metric("总售电均价", f"{summ['总售电均价(元/kWh)']:.4f}元")
    c7.metric("绿电消纳率", f"{summ['绿电消纳率']:.2%}")
    c8.metric("负荷满足率", f"{summ['负荷满足率']:.2%}")

    c13, c14, c15, c16 = st.columns(4)
    c13.metric("全年弃电量", f"{summ['全年总弃电量(MWh)']:.2f}MWh")
    c14.metric("全年缺口电量", f"{summ['全年总缺口电量(MWh)']:.2f}MWh")
    c15.metric(f"折扣后总收入（储能×{es_discount():.2f}）",
               f"{discounted_income(summ):.2f}万元")
    # 折后总售电均价（消纳口径自洽：折后总收入×10 ÷ 折后总消纳量）
    _d_now = es_discount()
    _recv_dir = (summ.get("光伏分项", {}).get("用户消纳电量(MWh)", 0.0)
                 + summ.get("风电分项", {}).get("用户消纳电量(MWh)", 0.0))
    _recv_es = summ.get("储能分项", {}).get("用户消纳电量(MWh)")
    _inc_disc = discounted_income(summ)
    if _recv_es is not None and _inc_disc is not None:
        _elec_disc = _recv_dir + _recv_es * _d_now
        _price_disc = _inc_disc * 10.0 / _elec_disc if _elec_disc > 1e-9 else 0.0
        c16.metric(f"折后总售电均价（×{_d_now:.2f}）", f"{_price_disc:.4f}元/kWh")
    else:
        c16.metric("折后总售电均价", "—")
    _es_rev = summ.get("储能放电收入(万元)")
    _pw_rev = summ.get("风光直供收入(万元)")
    if _es_rev is not None and _pw_rev is not None:
        _d_cur = es_discount()
        _dis_txt = (f"折后消纳量 {_recv_es * _d_cur:.2f} MWh（原 {_recv_es:.2f}）｜ "
                    if _recv_es is not None else "")
        st.caption(
            "折扣口径：储能用户消纳量打折，折后收入 = 折后消纳量 × 逐时电价｜ "
            + _dis_txt +
            f"风光直供收入 {_pw_rev:.2f} + 储能收入 {_es_rev:.2f}×{_d_cur:.2f}"
            f" = 折扣后总收入 {discounted_income(summ):.2f} 万元"
        )

    # 3. 分项统计
    st.markdown("")
    st.subheader("📂 风光储分项电量统计")
    col_pv, col_wt, col_es = st.columns(3)

    with col_pv:
        st.markdown("**☀️ 光伏分项**")
        pv_df = pd.DataFrame.from_dict(summ["光伏分项"], orient="index", columns=["数值"])
        pv_df["数值"] = pv_df["数值"].round(4)
        st.dataframe(pv_df, use_container_width=True)

    with col_wt:
        st.markdown("**🌬️ 风电分项**")
        wt_df = pd.DataFrame.from_dict(summ["风电分项"], orient="index", columns=["数值"])
        wt_df["数值"] = wt_df["数值"].round(4)
        st.dataframe(wt_df, use_container_width=True)

    with col_es:
        st.markdown("**🔋 储能分项**")
        es_df = es_detail_with_discount(summ["储能分项"])
        st.dataframe(es_df, use_container_width=True)

    # 5. 曲线与明细Tab
    st.markdown("")
    tab1, tab2, tab3 = st.tabs(["📈 运行曲线", "📋 逐时明细", "📥 下载结果"])

    with tab1:
        st.subheader("曲线展示设置")
        show_multi = st.checkbox("开启多方案叠加对比", value=False)
        if show_multi:
            selected_plot = st.multiselect("选择要叠加的方案", keys, default=[selected_key])
        else:
            selected_plot = [selected_key]

        sd, ed = pick_time_range()

        fig = plot_multi_results(results, selected_plot, sd, ed)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("8760逐时仿真明细")
        display_df = apply_es_discharge_discount(rdf.copy())
        display_df["时间"] = display_df["时间"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display_df, height=500, use_container_width=True)

    with tab3:
        st.subheader("下载测算结果")
        export_df = apply_es_discharge_discount(rdf.copy())
        export_df["时间"] = export_df["时间"].dt.strftime("%Y-%m-%d %H:%M")

        with pd.ExcelWriter("绿电直连项目仿真测算报表.xlsx") as writer:
            compare_df.to_excel(writer, sheet_name="方案对比汇总", index=False)
            sum_flat = {k: v for k, v in summ.items() if not isinstance(v, dict)}
            pd.DataFrame([sum_flat]).T.to_excel(writer, sheet_name="核心指标汇总", header=False)
            pd.DataFrame(summ["光伏分项"], index=["光伏"]).T.to_excel(writer, sheet_name="分项统计")
            pd.DataFrame(summ["风电分项"], index=["风电"]).T.to_excel(writer, sheet_name="分项统计", startrow=10)
            es_detail_with_discount(summ["储能分项"]).rename(
                columns={"数值": "储能"}).T.to_excel(writer, sheet_name="分项统计", startrow=20)
            export_df.to_excel(writer, sheet_name="8760逐时明细", index=False)

        download_file("📊 下载Excel测算报表", "绿电直连项目仿真测算报表.xlsx")

        fig_full = plot_multi_results(results, [selected_key])
        fig_full.write_html("仿真结果全周期曲线.html")
        download_file("📈 下载交互式图表(HTML)", "仿真结果全周期曲线.html",
                      mime="text/html")


# ===================== 主界面 =====================
def main():
    st.set_page_config(page_title="绿电直连时序仿真工具", page_icon="⚡", layout="wide")
    st.title("⚡ 绿电直连项目时序仿真工具")

    # =====================================================================
    # 侧边栏 = 全局配置
    # =====================================================================
    with st.sidebar:
        st.header("🧭 全局配置")

        # ===== LP求解器选择（CBC / HiGHS）=====
        st.subheader("🔧 LP求解器")
        _highs_available = "HiGHS" in pulp.listSolvers(onlyAvailable=True)
        if _highs_available:
            solver_choice = st.radio(
                "LP求解器",
                ["CBC (默认)", "HiGHS (更快)"],
                horizontal=True,
                help="CBC为PuLP自带；HiGHS为新一代开源求解器，大规模LP速度快数倍"
            )
        else:
            solver_choice = "CBC (默认)"
        lp_solver_name = "highs" if solver_choice.startswith("HiGHS") else "cbc"

        # ===== 储能放电收益折扣（展示层折后列） =====
        st.subheader("💰 收益折扣")
        st.number_input(
            "储能放电量折扣系数",
            min_value=0.0, max_value=1.0, value=ES_DISCOUNT_DEFAULT, step=0.05,
            format="%.2f", key="es_rev_discount",
            help="折扣对象为储能用户消纳电量（放电量×(1−线损)），与储能放电收入结算口径一致；"
                 "展示层始终同时给出原值与折后列。"
                 "折扣后总收入 = 风光直供收入 + 折后消纳电量×逐时电价。",
        )

        # ===== 高级设置（低频修改的环境项，默认折叠） =====
        with st.expander("⚙️ 高级设置", expanded=False):
            # ----- 电价扣减费用（按月） -----
            st.markdown("**💶 电价扣减费用（按月）**")
            default_sys = st.number_input("默认系统运行费(元/kWh)", 0.0, value=0.00, step=0.005)
            default_gov = st.number_input("默认政府基金附加(元/kWh)", 0.0, value=0.00, step=0.005)
            default_line_fee = st.number_input("默认线损费(元/kWh)", 0.0, value=0.00, step=0.005)

            if st.button("应用默认值到全部12个月", use_container_width=True):
                for m in range(1, 13):
                    st.session_state[f"sys_{m}"] = default_sys
                    st.session_state[f"gov_{m}"] = default_gov
                    st.session_state[f"linef_{m}"] = default_line_fee
                st.rerun()

            monthly_fees = {}
            for m in range(1, 13):
                st.markdown(f"**{m}月**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    sys_fee = st.number_input(f"系统运行费{m}月", value=default_sys, step=0.005, key=f"sys_{m}")
                with c2:
                    gov_fee = st.number_input(f"政府基金{m}月", value=default_gov, step=0.005, key=f"gov_{m}")
                with c3:
                    line_fee = st.number_input(f"线损费{m}月", value=default_line_fee, step=0.005, key=f"linef_{m}")
                monthly_fees[m] = (sys_fee, gov_fee, line_fee)

    # =====================================================================
    # 主区域：①数据 → ②任务参数 → 运行 → ③结果
    # =====================================================================

    # ===== ① 数据区 =====
    with st.container(border=True):
        st.subheader("① 数据")
        data = upload_and_validate_file()

    # ===== ② 任务参数区（装机/储能/专线/模型选择与运行按钮） =====
    with st.container(border=True):
        st.subheader("② 任务参数")

        st.markdown("**📦 装机配置**")
        pv_inst = st.number_input("光伏装机(kW)", 0.0, value=100000.0, step=10.0)
        wt_inst = st.number_input("风电装机(kW)", 0.0, value=100000.0, step=10.0)
        es_cap = st.number_input("储能容量(kWh)", 0.0, value=200000.0, step=10.0)

        st.markdown("**🔋 储能基础参数**")
        es_ce = st.number_input("充电效率", 0.0, 1.0, 0.92, 0.01)
        es_de = st.number_input("放电效率", 0.0, 1.0, 0.92, 0.01)
        soc_min = st.number_input("SOC下限", 0.0, 1.0, 0.05, 0.01)
        soc_max = st.number_input("SOC上限", 0.0, 1.0, 0.95, 0.01)
        soc0 = st.number_input("初始SOC", 0.0, 1.0, 0.5, 0.01)

        st.markdown("**🛰️ 专线参数**")
        line_length = st.number_input("专线长度(km)", 0.0, value=0.0, step=1.0)
        loss_per_km = st.number_input("单位公里线损率(%)", 0.0, value=0.0, step=0.05)
        line_loss_rate = line_length * loss_per_km / 100.0

        st.markdown("**🧠 运行模式选择**")
        st.subheader("🧠 仿真模型选择")

        model_labels = [m[0] for m in MODEL_OPTIONS]
        model_captions = {m[0]: m[2] for m in MODEL_OPTIONS}
        model_key_map = {m[0]: m[1] for m in MODEL_OPTIONS}

        selected_models = st.multiselect(
            "选择要运行的仿真模型（可多选，建议最多2个方便对比）",
            options=model_labels,
            default=[model_labels[0], model_labels[1]],
            format_func=lambda x: f"🧮 {x}",
        )

        if len(selected_models) == 0:
            st.warning("⚠️ 请至少选择 1 个仿真模型")

        # 显示每个选中模型的说明
        for m in selected_models:
            st.caption(f"  📌 **{m}**: {model_captions[m]}")

        # 求解超时时间（仅对 LP 有意义）
        needs_solver = any(
            model_key_map[m] in ["lp"] for m in selected_models
        )
        solver_time = None
        if needs_solver:
            solver_time = st.number_input("求解超时时间(秒)", 60, 6000, 3000, 30)

        # 储能充放时长（影响最大充放电功率 P_max = E / h）
        es_charge_hours = st.number_input(
            "储能充放时长(h)", 0.5, 24.0, 2.0, 0.5,
            help="储能最大充放电功率 = 容量 / 时长，2h 表示 200MWh 储能最大 100MW 功率")

        run_btn = st.button("🚀 运行仿真（单方案）", type="primary",
                            use_container_width=True, key="btn_single")

    # ===== 前台执行单方案仿真 =====
    if run_btn:
        if data is None:
            st.error("❌ 请先上传数据文件")
        elif not selected_models:
            st.error("❌ 请至少选择 1 个仿真模型")
        else:
            pv, wt, ld, pr_raw, time_idx = data
            net_price = preprocess_price(pr_raw, time_idx, monthly_fees)

            with st.spinner("⏳ 正在运行仿真..."):
                try:
                    results = {}
                    total_steps = len(selected_models)
                    start_time = time.time()
                    step = 0

                    progress_box = st.empty()

                    def update_progress(label):
                        nonlocal step
                        step += 1
                        elapsed = time.time() - start_time
                        pct = (step - 1) / max(total_steps, 1) * 100
                        progress_box.markdown(
                            f"⏳ **{label}** ｜ 步骤 **{step}/{total_steps}** ｜ "
                            f"已用 **{elapsed:.1f}s** ｜ 进度 **{pct:.0f}%**"
                        )

                    for model_label, model_key in zip(selected_models,
                                                      [model_key_map[m] for m in selected_models]):
                        update_progress(f"{model_label} 仿真中")

                        fn = MODEL_FN_MAP[model_key]

                        # 固定策略函数签名不同（不需要 solver_timeout）
                        if model_key == "fixed":
                            rdf, summ = fn(
                                pv_unit_power=pv, wt_unit_power=wt, load_profile=ld, net_price_profile=net_price,
                                pv_installed=pv_inst, wt_installed=wt_inst, es_capacity=es_cap,
                                es_charge_hours=es_charge_hours, es_charge_eff=es_ce, es_discharge_eff=es_de,
                                soc_min=soc_min, soc_max=soc_max,
                                line_loss_rate=line_loss_rate, soc_initial=soc0,
                            )
                        else:
                            rdf, summ = fn(
                                pv_unit_power=pv, wt_unit_power=wt, load_profile=ld, net_price_profile=net_price,
                                pv_installed=pv_inst, wt_installed=wt_inst, es_capacity=es_cap,
                                es_charge_hours=es_charge_hours, es_charge_eff=es_ce, es_discharge_eff=es_de,
                                soc_min=soc_min, soc_max=soc_max,
                                line_loss_rate=line_loss_rate, soc_initial=soc0, solver_timeout=solver_time,
                                solver_name=lp_solver_name,
                            )

                        results[model_label] = (rdf, summ)

                    elapsed_total = time.time() - start_time
                    progress_box.markdown(
                        f"✅ **全部计算完成** ｜ 步骤 **{step}/{total_steps}** ｜ "
                        f"总耗时 **{elapsed_total:.1f}s** ｜ 进度 **100%**"
                    )

                    st.session_state["run_mode"] = "multi_model"
                    st.session_state["results"] = results
                    # 参数快照：结果渲染必须用运行时的装机与线损，
                    # 而不是之后被其他操作覆盖的侧边栏当前值
                    st.session_state["single_params"] = {
                        "pv_inst": pv_inst, "wt_inst": wt_inst, "es_cap": es_cap,
                        "es_ce": es_ce, "es_de": es_de,
                        "soc_min": soc_min, "soc_max": soc_max, "soc0": soc0,
                        "line_loss_rate": line_loss_rate,
                    }
                    st.success(f"✅ {len(selected_models)} 个模型仿真完成！耗时 {elapsed_total:.1f}s")

                except Exception as ex:
                    st.error(f"❌ 运行失败：{type(ex).__name__}: {ex}")
                    import traceback
                    st.code(traceback.format_exc())

    # ===== ③ 结果区 =====
    with st.container(border=True):
        st.subheader("③ 结果")
        if "results" in st.session_state:
            render_results()
        else:
            st.info("👈 请在上方「① 数据」上传文件、"
                    "「② 任务参数」填好装机与储能后点击运行按钮")


if __name__ == "__main__":
    main()
