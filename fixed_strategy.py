import numpy as np
import pandas as pd

# ===================== 固定策略仿真（考虑专线线损） =====================
def fixed_strategy_simulation(
    pv_unit_power, wt_unit_power, load_profile, net_price_profile,
    pv_installed, wt_installed, es_capacity,
    es_charge_hours, es_charge_eff, es_discharge_eff,
    soc_min=0.1, soc_max=1.0, line_loss_rate=0.05, soc_initial=0.5,
    es_dod=None,  # 向后兼容：老调用方仍传 es_dod，这里会反推 soc_min
):
    """
    固定策略：风光优先供电，盈余充电，不足放电；电站侧弃电，避免无效线损

    SOC 边界：
      - soc_min / soc_max 为直接输入的上下限（推荐方式）
      - 若传了 es_dod 而 soc_min 为 None，则 soc_min = 1 - es_dod（兼容旧调用）

    line_loss_rate: 专线总损耗率（小数，如0.05代表5%）
    """
    pv_unit_power = np.asarray(pv_unit_power, dtype=np.float64).flatten()
    wt_unit_power = np.asarray(wt_unit_power, dtype=np.float64).flatten()
    load_profile = np.asarray(load_profile, dtype=np.float64).flatten()
    net_price_profile = np.asarray(net_price_profile, dtype=np.float64).flatten()
    
    hour_count = 8760
    for name, arr in [("光伏出力", pv_unit_power), ("风电出力", wt_unit_power),
                      ("用户负荷", load_profile), ("净电价", net_price_profile)]:
        if arr.shape[0] != hour_count:
            raise ValueError(f"{name}数据长度错误：当前{arr.shape[0]}，必须为8760")
    
    es_capacity = float(es_capacity)
    es_charge_hours = float(es_charge_hours)
    has_es = es_capacity > 0 and es_charge_hours > 0
    es_power = es_capacity / es_charge_hours if has_es else 0.0

    # 兼容处理：soc_min/soc_max 没显式传时，从 es_dod 反推（老调用方传 es_dod）
    if soc_min is None and es_dod is not None:
        soc_min = 1.0 - float(es_dod)
    if soc_max is None:
        soc_max = 1.0
    soc_min = float(soc_min)
    soc_max = float(soc_max)

    # 前置参数校验：soc_initial 必须在 [soc_min, soc_max] 内
    if has_es and not (soc_min - 1e-9 <= soc_initial <= soc_max + 1e-9):
        raise ValueError(
            f"参数矛盾：初始SOC={soc_initial:.3f} 不在允许区间 "
            f"[{soc_min:.3f}, {soc_max:.3f}] 内。"
            f"请调整 SOC上下限 或 初始SOC"
        )
    
    # 初始化数组
    pv_gen = np.zeros(hour_count, dtype=np.float64)
    wt_gen = np.zeros(hour_count, dtype=np.float64)
    total_gen = np.zeros(hour_count, dtype=np.float64)
    es_charge = np.zeros(hour_count, dtype=np.float64)
    es_discharge = np.zeros(hour_count, dtype=np.float64)
    soc = np.zeros(hour_count, dtype=np.float64)
    abandon_power = np.zeros(hour_count, dtype=np.float64)
    shortage_power = np.zeros(hour_count, dtype=np.float64)
    green_power_used = np.zeros(hour_count, dtype=np.float64)
    line_loss = np.zeros(hour_count, dtype=np.float64)
    send_end_power = np.zeros(hour_count, dtype=np.float64)
    hour_income = np.zeros(hour_count, dtype=np.float64)
    
    soc[0] = float(soc_initial) if has_es else 0.0
    
    for t in range(hour_count):
        pv_gen[t] = float(pv_installed) * pv_unit_power[t]
        wt_gen[t] = float(wt_installed) * wt_unit_power[t]
        total_gen[t] = pv_gen[t] + wt_gen[t]
        
        if not has_es:
            # 无储能场景
            max_receive = total_gen[t] * (1 - line_loss_rate)
            if max_receive >= load_profile[t]:
                green_power_used[t] = load_profile[t]
                needed_send = load_profile[t] / (1 - line_loss_rate)
                send_end_power[t] = needed_send
                abandon_power[t] = total_gen[t] - needed_send
                shortage_power[t] = 0.0
            else:
                green_power_used[t] = max_receive
                send_end_power[t] = total_gen[t]
                shortage_power[t] = load_profile[t] - max_receive
                abandon_power[t] = 0.0
            line_loss[t] = send_end_power[t] * line_loss_rate
        else:
            # 有储能固定策略
            max_receive_gen = total_gen[t] * (1 - line_loss_rate)
            
            if max_receive_gen >= load_profile[t]:
                # 风光盈余，优先充电
                needed_send = load_profile[t] / (1 - line_loss_rate)
                surplus_gen = total_gen[t] - needed_send
                
                max_chg = min((soc_max - soc[t]) * es_capacity / float(es_charge_eff), es_power)
                es_charge[t] = min(surplus_gen, max_chg)
                abandon_power[t] = surplus_gen - es_charge[t]
                es_discharge[t] = 0.0
                
                send_end_power[t] = needed_send
                green_power_used[t] = load_profile[t]
                shortage_power[t] = 0.0
            else:
                # 风光不足，放电补充
                shortage_receive = load_profile[t] - max_receive_gen
                needed_dis = shortage_receive / (1 - line_loss_rate)
                
                max_dis = min((soc[t] - soc_min) * es_capacity * float(es_discharge_eff), es_power)
                es_discharge[t] = min(needed_dis, max_dis)
                es_charge[t] = 0.0
                abandon_power[t] = 0.0
                
                total_receive = (total_gen[t] + es_discharge[t]) * (1 - line_loss_rate)
                if total_receive >= load_profile[t]:
                    green_power_used[t] = load_profile[t]
                    send_end_power[t] = load_profile[t] / (1 - line_loss_rate)
                    shortage_power[t] = 0.0
                else:
                    green_power_used[t] = total_receive
                    send_end_power[t] = total_gen[t] + es_discharge[t]
                    shortage_power[t] = load_profile[t] - total_receive
            
            line_loss[t] = send_end_power[t] * line_loss_rate
            
            # 更新SOC
            chg_e = es_charge[t] * float(es_charge_eff)
            dis_e = es_discharge[t] / float(es_discharge_eff)
            if t < hour_count - 1:
                nxt = soc[t] + (chg_e - dis_e) / es_capacity
                soc[t + 1] = max(soc_min, min(soc_max, nxt))
        
        hour_income[t] = green_power_used[t] * net_price_profile[t]
    
    # 构建结果表
    time_idx = pd.date_range(start="2026-01-01 00:00:00", periods=8760, freq="h")
    es_power_net = es_charge - es_discharge

    # ===== 逐时发电比例归因：风光储消纳、风光弃电 =====
    # 每小时按当小时风光出力占比拆分直供消纳与弃电；储能消纳=放电×(1−线损)
    pv_share_t = np.divide(pv_gen, total_gen, out=np.zeros_like(pv_gen), where=total_gen > 0)
    direct_send_t = np.maximum(send_end_power - es_discharge, 0.0)
    pv_consume_t = direct_send_t * (1 - line_loss_rate) * pv_share_t
    wt_consume_t = direct_send_t * (1 - line_loss_rate) * (1 - pv_share_t)
    es_consume_t = es_discharge * (1 - line_loss_rate)
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
        "储能SOC": soc,
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
    
    # 汇总统计 + 分项拆分
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
    # 仿真与经济性计算均不经过此段，仅用于报表归因展示。
    pv_ratio = pv_gen.sum() / tg if tg > 0 else 0.0
    wt_ratio = 1.0 - pv_ratio
    total_send = send_end_power.sum()
    total_direct_send = total_send - total_discharge

    pv_detail = {
        "理论发电量(MWh)": pv_gen.sum() / 1000.0,
        "直供送端电量(MWh)": total_direct_send * pv_ratio / 1000.0,
        "进储能电量(MWh)": total_charge * pv_ratio / 1000.0,
        "弃电量(MWh)": abandon_power.sum() * pv_ratio / 1000.0,
        "专线折损量(MWh)": total_direct_send * pv_ratio * line_loss_rate / 1000.0,
        "用户消纳电量(MWh)": total_direct_send * pv_ratio * (1 - line_loss_rate) / 1000.0,
    }
    wt_detail = {
        "理论发电量(MWh)": wt_gen.sum() / 1000.0,
        "直供送端电量(MWh)": total_direct_send * wt_ratio / 1000.0,
        "进储能电量(MWh)": total_charge * wt_ratio / 1000.0,
        "弃电量(MWh)": abandon_power.sum() * wt_ratio / 1000.0,
        "专线折损量(MWh)": total_direct_send * wt_ratio * line_loss_rate / 1000.0,
        "用户消纳电量(MWh)": total_direct_send * wt_ratio * (1 - line_loss_rate) / 1000.0,
    }
    
    # 储能分项
    es_loss_charge = total_charge * (1 - float(es_charge_eff))
    es_loss_discharge = total_discharge * (1.0 / float(es_discharge_eff) - 1.0)
    es_detail = {
        "总充电量(MWh)": total_charge / 1000.0,
        "总放电量(MWh)": total_discharge / 1000.0,
        "充电内部损耗(MWh)": es_loss_charge / 1000.0,
        "放电内部损耗(MWh)": es_loss_discharge / 1000.0,
        "储能内部总损耗(MWh)": (es_loss_charge + es_loss_discharge) / 1000.0,
        "放电侧专线折损(MWh)": total_discharge * line_loss_rate / 1000.0,
        "用户消纳电量(MWh)": total_discharge * (1 - line_loss_rate) / 1000.0,
        "最大SOC": float(soc.max()) if has_es else 0.0,
        "最小SOC": float(soc.min()) if has_es else 0.0,
        "平均SOC": float(soc.mean()) if has_es else 0.0,
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
        "策略类型": "固定策略",
        "充放时长(h)": es_charge_hours,
    }
    return result_df, summary
