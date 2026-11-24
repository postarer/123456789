import numpy as np
import pandas as pd
import pulp

# ===================== LP松弛策略仿真（去掉充放电互斥约束） =====================
def lp_strategy_simulation(
    pv_unit_power, wt_unit_power, load_profile, net_price_profile,
    pv_installed, wt_installed, es_capacity,
    es_charge_hours, es_charge_eff, es_discharge_eff,
    soc_min=0.1, soc_max=1.0, line_loss_rate=0.05, soc_initial=0.5, solver_timeout=300,
    solver_name="cbc", solver_msg=1,
    es_dod=None,  # 向后兼容
):
    """
    LP线性规划：最大化全年总收入（无充放电互斥约束）
    去掉 MILP 中的二进制变量 u[t] 和两条互斥约束，所有变量连续
    理论上效率损失使最优解不会同时充放电，但退化解可能出现
    求解器：CBC（默认，PuLP内置）或 HiGHS（solver_name="highs"，需装 highspy）

    SOC 边界：soc_min/soc_max 为直接输入，若未传且给了 es_dod 则反推 soc_min=1-es_dod
    """
    pv_unit_power = np.asarray(pv_unit_power, dtype=np.float64).flatten()
    wt_unit_power = np.asarray(wt_unit_power, dtype=np.float64).flatten()
    load_profile = np.asarray(load_profile, dtype=np.float64).flatten()
    net_price_profile = np.asarray(net_price_profile, dtype=np.float64).flatten()

    T = len(pv_unit_power)
    for name, arr in [("光伏出力", pv_unit_power), ("风电出力", wt_unit_power),
                      ("用户负荷", load_profile), ("净电价", net_price_profile)]:
        if len(arr) != T:
            raise ValueError(f"{name}数据长度错误：当前{len(arr)}，必须与光伏出力一致({T})")
    pv_gen = pv_installed * pv_unit_power
    wt_gen = wt_installed * wt_unit_power
    total_gen = pv_gen + wt_gen

    es_capacity = float(es_capacity)
    es_charge_hours = float(es_charge_hours)
    P_max = es_capacity / es_charge_hours if es_capacity > 0 else 0.0

    # 兼容处理：soc_min/soc_max 没显式传时，从 es_dod 反推（老调用方传 es_dod）
    if soc_min is None and es_dod is not None:
        soc_min = 1.0 - float(es_dod)
    if soc_max is None:
        soc_max = 1.0
    soc_min = float(soc_min)
    soc_max = float(soc_max)

    eta_c = float(es_charge_eff)
    eta_d = float(es_discharge_eff)
    eta_line = float(line_loss_rate)

    # 安全分母：容量为 0 时 P_max=0 强制充放功率恒为 0，
    # 分子恒为 0，分母取何值都不影响约束，仅避免构造约束时除零
    safe_es_capacity = es_capacity if es_capacity > 0 else 1.0

    # 前置参数校验：避免约束互相矛盾导致求解器报 Infeasible
    if not (soc_min - 1e-9 <= soc_initial <= soc_max + 1e-9):
        raise ValueError(
            f"参数矛盾：初始SOC={soc_initial:.3f} 不在允许区间 "
            f"[{soc_min:.3f}, {soc_max:.3f}] 内。"
            f"请调整 SOC上下限 或 初始SOC"
        )

    # 构建优化问题
    prob = pulp.LpProblem("GreenPower_ES_LP", pulp.LpMaximize)

    # 变量定义（全部连续，无二进制变量）
    charge = pulp.LpVariable.dicts("charge", range(T), lowBound=0, upBound=P_max)
    discharge = pulp.LpVariable.dicts("discharge", range(T), lowBound=0, upBound=P_max)
    soc = pulp.LpVariable.dicts("soc", range(T), lowBound=soc_min, upBound=soc_max)
    green_used = pulp.LpVariable.dicts("green_used", range(T), lowBound=0)
    abandon = pulp.LpVariable.dicts("abandon", range(T), lowBound=0)

    # 目标函数：全年净收入最大化 + SOC微奖励（打破多解平局，偏好早充）
    # LP去掉互斥约束后解空间更大，持续弃电窗口内"早充/晚充"收入完全等价
    # 需要微小SOC奖励把解推向"持续弃电时优先充满"的工程合理行为
    # 权重1e-4远小于电价(~0.4)，不改变"该充/不该充"的根本决策，只打破平局
    soc_bonus_weight = 1e-2
    prob += pulp.lpSum([green_used[t] * net_price_profile[t] + soc_bonus_weight * soc[t] for t in range(T)])

    # 约束条件（无充放电互斥约束）
    for t in range(T):
        # 1. 电量平衡：送端 = 总发电 - 充电 - 弃电 + 放电；受端 = 送端*(1-线损) = 绿电消纳
        prob += total_gen[t] - charge[t] - abandon[t] + discharge[t] == green_used[t] * (1.0 / (1.0 - eta_line))

        # 2. 绿电消纳不超过用户负荷
        prob += green_used[t] <= load_profile[t]

        # 3. SOC动态方程
        if t == 0:
            prob += soc[t] == soc_initial
        else:
            prob += soc[t] == soc[t-1] + (charge[t-1] * eta_c - discharge[t-1] * (1.0 / eta_d)) * (1.0 / safe_es_capacity)

    # 求解（CBC 默认；HiGHS 更快）
    if solver_name.lower() == "highs":
        solver = pulp.HiGHS(msg=solver_msg, timeLimit=solver_timeout)
    else:
        solver = pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=solver_timeout)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] not in ["Optimal", "Feasible"]:
        raise RuntimeError(f"LP求解失败，状态：{pulp.LpStatus[prob.status]}")

    # 提取结果
    es_charge = np.array([charge[t].varValue for t in range(T)], dtype=np.float64)
    es_discharge = np.array([discharge[t].varValue for t in range(T)], dtype=np.float64)
    soc_arr = np.array([soc[t].varValue for t in range(T)], dtype=np.float64)
    green_power_used = np.array([green_used[t].varValue for t in range(T)], dtype=np.float64)
    abandon_power = np.array([abandon[t].varValue for t in range(T)], dtype=np.float64)

    # ===== 后处理：若出现同时充放电，取 min 清零（LP 松弛退化解可能有此问题）=====
    both_mask = (es_charge > 1e-3) & (es_discharge > 1e-3)
    if both_mask.any():
        both_idx = np.where(both_mask)[0]
        for t in both_idx:
            overlap = min(es_charge[t], es_discharge[t])
            es_charge[t] -= overlap
            es_discharge[t] -= overlap
        print(f"[LP后处理] 发现 {len(both_idx)} 个时刻同时充放电，已取min清零")

        # 清零后充放功率与 SOC 不再自洽，需重新滚动计算
        # 关键账本修复：SOC 顶格时"充不进"的电量必须从充电功率中扣除并等量转记弃电。
        # 原理：环流清洗把原本互相抵消的充电量变成了真实充电，可能把 SOC 顶过上限；
        # 若只钳制 SOC 而不减充电功率，会出现"满电还在充"的物理不可能状态，
        # 充入电量凭空消失，导致储能分项能量不守恒（ΣC×ηc − ΣD/ηd ≠ ΔSOC×E_cap）
        # 转记弃电后：功率平衡中充电与弃电同侧（send = gen − charge − abandon + discharge 不变），
        # 送端功率、线损、消纳量、收入全部保持不变，仅修正储能分项的统计口径
        soc_arr = np.zeros(T, dtype=np.float64)
        soc_arr[0] = soc_initial
        overflow_total = 0.0  # 触顶转弃电的累计充电功率（送端口径 kWh）
        for t in range(T - 1):
            raw_next = soc_arr[t] + (es_charge[t] * eta_c - es_discharge[t] / eta_d) / safe_es_capacity
            if raw_next > soc_max + 1e-12:
                # 本小时想冲入的电池内电量超过剩余容量空间：
                # 超出量（电池内口径）= (raw_next − soc_max)×E_cap，折回送端充电功率需除以 ηc
                excess_e = (raw_next - soc_max) * safe_es_capacity
                excess_charge = excess_e / eta_c
                es_charge[t] -= excess_charge      # 削减充不进的充电功率
                abandon_power[t] += excess_charge  # 等量转记弃电（电确实发了但没进电池）
                overflow_total += excess_charge
                soc_arr[t + 1] = soc_max
            else:
                soc_arr[t + 1] = max(soc_min, min(soc_max, raw_next))
        if overflow_total > 1e-3:
            print(f"[LP后处理] SOC顶格时段充不进的电量 {overflow_total/1000:.2f} MWh 已转记弃电")

    # ===== 末小时账本修复：消除 charge[T-1] 游离变量的统计黑洞 =====
    # SOC 递推只建到 soc[T-1]（由 charge[T-2] 决定），末小时 charge[T-1] 不受任何 SOC 约束；
    # 而弃电严重时段充电与弃电在目标函数中系数同为 0（完全平局），
    # LP 可能把本该弃掉的电记成"充电"——账面充电量虚高且能量不守恒。
    # 物理上末小时可充入的电量受电池剩余容量限制，超出部分同样转记弃电。
    # （纯记账修正：转记后功率平衡、收入、消纳均不变，末小时 SOC 已在顶格时全额转弃电）
    room_e = (soc_max - soc_arr[T - 1]) * safe_es_capacity  # 电池内剩余空间(kWh)
    max_end_charge = room_e / eta_c                          # 折算送端充电功率上限(kW)
    if es_charge[T - 1] > max_end_charge + 1e-9:
        excess_end = es_charge[T - 1] - max_end_charge
        es_charge[T - 1] = max_end_charge
        abandon_power[T - 1] += excess_end
        print(f"[LP后处理] 末小时充电超出电池剩余空间 {excess_end/1000:.2f} MWh，已转记弃电")

    send_end_power = total_gen - es_charge - abandon_power + es_discharge
    line_loss = send_end_power * eta_line
    shortage_power = np.maximum(load_profile - green_power_used, 0.0)
    hour_income = green_power_used * net_price_profile

    # 构建结果表
    time_idx = pd.date_range(start="2026-01-01 00:00:00", periods=T, freq="h")
    es_power_net = es_charge - es_discharge

    # ===== 逐时发电比例归因：风光储消纳、风光弃电 =====
    # 每小时按当小时风光出力占比拆分直供消纳与弃电；储能消纳=放电×(1−线损)
    pv_share_t = np.divide(pv_gen, total_gen, out=np.zeros_like(pv_gen), where=total_gen > 0)
    direct_send_t = np.maximum(send_end_power - es_discharge, 0.0)
    pv_consume_t = direct_send_t * (1 - eta_line) * pv_share_t
    wt_consume_t = direct_send_t * (1 - eta_line) * (1 - pv_share_t)
    es_consume_t = es_discharge * (1 - eta_line)
    pv_abandon_t = abandon_power * pv_share_t
    wt_abandon_t = abandon_power * (1 - pv_share_t)

    result_df = pd.DataFrame({
        "时间": time_idx,
        "光伏发电量(kW)": pv_gen,
        "风电发电量(kW)": wt_gen,
        "风光总发电量(kW)": total_gen,
        "用户负荷(kW)": load_profile,
        "储能充电功率(kW)": es_charge,
        "储能放电功率(kW)": es_discharge,
        "储能功率(kW)": es_power_net,
        "储能SOC": soc_arr,
        "专线送端功率(kW)": send_end_power,
        "专线损耗功率(kW)": line_loss,
        "弃电功率(kW)": abandon_power,
        "缺口功率(kW)": shortage_power,
        "绿电消纳量(kWh)": green_power_used,
        "光伏消纳量(kWh)": pv_consume_t,
        "风电消纳量(kWh)": wt_consume_t,
        "储能消纳量(kWh)": es_consume_t,
        "光伏弃电量(kWh)": pv_abandon_t,
        "风电弃电量(kWh)": wt_abandon_t,
        "净电价(元/kWh)": net_price_profile,
        "当期收入(元)": hour_income
    })

    # 汇总统计（同固定策略 / MILP 口径）
    tg = total_gen.sum()
    tl = load_profile.sum()
    total_income = hour_income.sum()
    total_green_mwh = green_power_used.sum() / 1000.0
    total_line_loss_mwh = line_loss.sum() / 1000.0
    avg_price = (total_income / (total_green_mwh * 1000)) if total_green_mwh > 0 else 0.0

    total_charge = es_charge.sum()
    total_discharge = es_discharge.sum()

    # ===== 分项统计：按全年发电量占比比例分摊（汇报口径，不影响任何计算与经济性）=====
    # 物理上母线电流不可区分，"哪度电是风/光的"只是统计约定。
    # 本口径将直供/进储能/弃电/折损各总量按风、光全年发电量占比统一拆分：
    #   - 各分项加总严格等于总量，账面自洽
    #   - 分电源弃电率与总弃电率相同（比例分摊口径的固有特征）
    # 求解与经济性计算均不经过此段，仅用于报表归因展示。
    pv_ratio = pv_gen.sum() / tg if tg > 0 else 0.0
    wt_ratio = 1.0 - pv_ratio
    total_send = send_end_power.sum()
    total_direct_send = total_send - total_discharge

    pv_detail = {
        "理论发电量(MWh)": pv_gen.sum() / 1000.0,
        "直供送端电量(MWh)": total_direct_send * pv_ratio / 1000.0,
        "进储能电量(MWh)": total_charge * pv_ratio / 1000.0,
        "弃电量(MWh)": abandon_power.sum() * pv_ratio / 1000.0,
        "专线折损量(MWh)": total_direct_send * pv_ratio * eta_line / 1000.0,
        "用户消纳电量(MWh)": total_direct_send * pv_ratio * (1 - eta_line) / 1000.0,
    }
    wt_detail = {
        "理论发电量(MWh)": wt_gen.sum() / 1000.0,
        "直供送端电量(MWh)": total_direct_send * wt_ratio / 1000.0,
        "进储能电量(MWh)": total_charge * wt_ratio / 1000.0,
        "弃电量(MWh)": abandon_power.sum() * wt_ratio / 1000.0,
        "专线折损量(MWh)": total_direct_send * wt_ratio * eta_line / 1000.0,
        "用户消纳电量(MWh)": total_direct_send * wt_ratio * (1 - eta_line) / 1000.0,
    }

    es_loss_charge = total_charge * (1 - eta_c)
    es_loss_discharge = total_discharge * (1.0 / eta_d - 1.0)
    es_detail = {
        "总充电量(MWh)": total_charge / 1000.0,
        "总放电量(MWh)": total_discharge / 1000.0,
        "充电内部损耗(MWh)": es_loss_charge / 1000.0,
        "放电内部损耗(MWh)": es_loss_discharge / 1000.0,
        "储能内部总损耗(MWh)": (es_loss_charge + es_loss_discharge) / 1000.0,
        "放电侧专线折损(MWh)": total_discharge * eta_line / 1000.0,
        "用户消纳电量(MWh)": total_discharge * (1 - eta_line) / 1000.0,
        "最大SOC": float(soc_arr.max()),
        "最小SOC": float(soc_arr.min()),
        "平均SOC": float(soc_arr.mean()),
    }

    # ===== 售电均价（逐时加权，与逐时发电比例归因口径一致）=====
    # pv_direct_t / wt_direct_t = 逐时风光直售送端量（上方已按 pv_share_t 拆分）
    # 同一小时风光同价，但两者逐时出力分布互补，加权后全年均价不同
    # 加权均价 = Σ(功率(t) × 电价(t)) / Σ功率(t)
    pv_direct_t = direct_send_t * pv_share_t
    wt_direct_t = direct_send_t - pv_direct_t

    pv_dir_sum = pv_direct_t.sum()
    wt_dir_sum = wt_direct_t.sum()
    es_dis_sum = es_discharge.sum()
    pv_direct_avg_price = ((pv_direct_t * net_price_profile).sum() / pv_dir_sum) if pv_dir_sum > 0 else 0.0
    wt_direct_avg_price = ((wt_direct_t * net_price_profile).sum() / wt_dir_sum) if wt_dir_sum > 0 else 0.0
    es_dis_avg_price = ((es_discharge * net_price_profile).sum() / es_dis_sum) if es_dis_sum > 0 else 0.0
    # 总售电均价 = 总收入 / 绿电消纳量（受端口径，线损因子约掉）
    total_sell_avg_price = (total_income / green_power_used.sum()) if green_power_used.sum() > 0 else 0.0

    summary = {
        "全年总发电量(MWh)": tg / 1000.0,
        "全年总绿电消纳量(MWh)": total_green_mwh,
        "全年专线总损耗(MWh)": total_line_loss_mwh,
        "全年总弃电量(MWh)": abandon_power.sum() / 1000.0,
        "全年总缺口电量(MWh)": shortage_power.sum() / 1000.0,
        # 消纳率 = 1 − 弃电率 = (理论发电量 − 弃电量) / 理论发电量
        # 送端统一口径，只反映"发出来的电有多少被浪费"，不含专线线损
        "绿电消纳率": ((tg - abandon_power.sum()) / tg) if tg > 0 else 0.0,
        "负荷满足率": (green_power_used.sum() / tl) if tl > 0 else 0.0,
        "全年总收入(万元)": total_income / 10000.0,
        "储能放电收入(万元)": float((es_consume_t * net_price_profile).sum()) / 10000.0,
        "风光直供收入(万元)": float(total_income - (es_consume_t * net_price_profile).sum()) / 10000.0,
        "消纳绿电平均净电价(元/kWh)": avg_price,
        "光伏直售均价(元/kWh)": pv_direct_avg_price,
        "风电直售均价(元/kWh)": wt_direct_avg_price,
        "储能放电均价(元/kWh)": es_dis_avg_price,
        "总售电均价(元/kWh)": total_sell_avg_price,
        "光伏分项": pv_detail,
        "风电分项": wt_detail,
        "储能分项": es_detail,
        "策略类型": "LP松弛策略",
        "充放时长(h)": es_charge_hours,
        "求解状态": pulp.LpStatus[prob.status],
    }
    return result_df, summary
