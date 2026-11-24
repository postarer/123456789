import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ===================== 多方案可视化绘图：两行标注、自动内外位置切换，防卡顿，标注单位万元 =====================
def plot_multi_results(result_dict, selected_keys, start_date=None, end_date=None, bar_threshold=200):
    """
    多方案叠加绘图
    :param result_dict: key=方案名, value=(result_df, summary)
    :param selected_keys: 选中要展示的方案列表
    :param start_date: 筛选起始时间
    :param end_date: 筛选结束时间
    :param bar_threshold: 筛选后数据行数 <=该阈值，才启用柱状图；大于阈值用折线，避免大量bar卡死
    """
    if not selected_keys:
        return go.Figure()
    # 【物理量固定颜色】 同一个物理量永远同一个颜色，跨方案不变
    color_map = {
        "风光总发电": "#2196F3",
        "用户负荷": "#4B4847",
        "储能功率": "#9C27B0",
        "储能SOC": "#167019",
        "逐时收入": "#92A213",
        "净电价": "#B53F45",
        "绿电消纳量": "#00ACC1",
        "弃电功率": "#FF9800",     # 橙色
        "缺口功率": "#D32F2F"      # ← 新增：红色
    }
    # 【不同方案使用不同线型，颜色复用上面color_map】
    line_styles = [
        dict(dash="solid"),     # 实线
        dict(dash="dash"),      # 虚线
        dict(dash="dot"),       # 点线
        dict(dash="dashdot"),   # 点划线
        dict(dash="longdash"),
        dict(dash="longdashdot")
    ]

    def auto_sparse_label_wan(yuan_series, max_label_cnt=30, min_wan=0.001):
        """
        自适应稀疏标签，输出【两行】：数值<br>万，小于阈值返回空字符串
        :param yuan_series: pd.Series 原始单位元
        :param max_label_cnt:最大标签数量
        :param min_wan:最小显示的万元阈值
        :return:text list
        """
        n = len(yuan_series)
        texts = [""] * n
        if n <= 0:
            return texts
        step = max(1, int(np.ceil(n / max_label_cnt)))
        for i in range(0, n, step):
            val_yuan = float(yuan_series.iloc[i])
            val_wan = val_yuan / 10000.0
            if val_wan >= min_wan:
                texts[i] = f"{val_wan:.1f}<br>万"   # <br>换行，两行文字
            else:
                texts[i] = ""
        return texts

    d_list = {}
    use_bar_mode = False
    for k in selected_keys:
        d = result_dict[k][0].copy()
        if start_date and end_date:
            mask = (d["时间"] >= pd.Timestamp(start_date)) & (d["时间"] <= pd.Timestamp(end_date))
            d = d[mask].reset_index(drop=True)
        d_list[k] = d
    # 判断是否启用柱状模式：取第一个方案筛选后行数做判断
    sample_df = d_list[selected_keys[0]]
    n_points = len(sample_df)
    if n_points <= bar_threshold:
        use_bar_mode = True

        # 统一坐标范围
    all_max = 0
    all_min = 0          # ← 新增：追踪实际最小值
    all_income_max = 0
    all_green_consume_max = 0
    all_price_max = 0    # ← 新增：追踪净电价最大值
    
    for k in selected_keys:
        d = d_list[k]
        pmax = max(
            d["风光总发电量(kW)"].max(),
            d["用户负荷(kW)"].max(),
            d["弃电功率(kW)"].max(),
            d["缺口功率(kW)"].max(),
            d["储能功率(kW)"].abs().max(),
        )
        # 储能功率画图时取了负号，必须单独算负向极值
        pmin = min(
            (-d["储能功率(kW)"]).min(),   # 取反后的最小值（最大充电功率的负数）
            0
        )
        all_max = max(all_max, pmax)
        all_min = min(all_min, pmin)
        all_income_max = max(all_income_max, d["当期收入(元)"].max())
        all_green_consume_max = max(all_green_consume_max, d["绿电消纳量(kWh)"].max())
        all_price_max = max(all_price_max, d["净电价(元/kWh)"].max())

    power_y_max = all_max * 1.15
    power_y_min = all_min * 1.15   # ← 负方向同样留 15% 边距，不再写死 -0.5
    # 柱状模式预留顶部空间；折线模式不需要额外留白
    if use_bar_mode:
        left3_y_max = max(all_green_consume_max, all_income_max) * 1.35   # 加大预留，防止外部两行文字被裁切
    else:
        left3_y_max = max(all_green_consume_max, all_income_max) * 1.1

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[2, 0.5, 1],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]]
    )

    for idx, k in enumerate(selected_keys):
        d = d_list[k]
        ls = line_styles[idx % len(line_styles)]
        suffix = f"【{k}】"

        # 子图1 row1：功率
        fig.add_trace(go.Scatter(
            x=d["时间"], y=d["风光总发电量(kW)"],
            name=f"风光总发电{suffix}",
            line=dict(color=color_map["风光总发电"], width=3.0, **ls)
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=d["时间"], y=d["用户负荷(kW)"],
            name=f"用户负荷{suffix}",
            line=dict(color=color_map["用户负荷"], width=3.0, **ls), opacity=0.8
        ), row=1, col=1, secondary_y=False)

        # 放电部分（>=0，实线，包含0）
        fig.add_trace(go.Scatter(
            x=d["时间"], y=-d["储能功率(kW)"],
            name=f"储能功率（正放负充）{suffix}",
            line=dict(color=color_map["储能功率"], width=3.0, **ls)
        ), row=1, col=1, secondary_y=False)

        # 弃电功率
        fig.add_trace(go.Scatter(
            x=d["时间"], y=d["弃电功率(kW)"],
            name=f"弃电功率{suffix}",
            line=dict(color=color_map["弃电功率"], width=3, **ls),
            opacity=0.7
        ), row=1, col=1, secondary_y=False)

        # 缺口功率（负荷减去新能源和储能出力后需额外从电网购买的功率）
        fig.add_trace(go.Scatter(
            x=d["时间"], y=d["缺口功率(kW)"],
            name=f"缺口功率{suffix}",
            line=dict(color=color_map["缺口功率"], width=3, **ls),
            opacity=0.7
        ), row=1, col=1, secondary_y=False)

        # 子图2 row2：SOC
        fig.add_trace(go.Scatter(
            x=d["时间"],
            y=d["储能SOC"],
            name=f"储能SOC{suffix}",
            line=dict(color=color_map["储能SOC"], width=2, **ls),
            fill="tozeroy",
            fillcolor="rgba(76, 175, 80, 0.25)",
            hovertemplate="%{y:.1%}"
        ), row=2, col=1)

        # ==========子图3 row3 ==========
        # 1.绿电消纳量：折线，左轴，永远绘制
        fig.add_trace(go.Scatter(
            x=d["时间"],
            y=d["绿电消纳量(kWh)"],
            name=f"绿电消纳量{suffix}",
            line=dict(color=color_map["绿电消纳量"], width=3, **ls),
            hovertemplate="时间:%{x}<br>绿电消纳量:%{y:.1f} kWh<extra></extra>"
        ), row=3, col=1, secondary_y=False)

                # 逐时收入：永远折线，不再使用柱状图
        fig.add_trace(go.Scatter(
            x=d["时间"],
            y=d["当期收入(元)"],
            name=f"逐时收入{suffix}",
            line=dict(color=color_map["逐时收入"], width=3, **ls),
            hovertemplate="时间:%{x}<br>逐时收入:%{y:.1f} 元<extra></extra>"
        ), row=3, col=1, secondary_y=False)


    # 净电价：全部方案一致，画在row3右轴 secondary_y=True
    first_d = d_list[selected_keys[0]]
    fig.add_trace(go.Scatter(
        x=first_d["时间"],
        y=first_d["净电价(元/kWh)"],
        name="净电价",
        line=dict(color=color_map["净电价"], width=3, dash="solid"),
        hovertemplate="时间:%{x}<br>净电价:%{y:.3f} 元/kWh<extra></extra>"
    ), row=3, col=1, secondary_y=True)

    fig.update_layout(
        height=950, hovermode="x unified", template="plotly_white",
        title="多方案仿真运行曲线对比",
        title_x=0.5,
        title_y=0.99,
        legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="center", x=0.5,
                    itemwidth=120, tracegroupgap=10, font=dict(size=15)),
        barmode="overlay"
    )

    fig.update_yaxes(title_text="功率(kW)", row=1, col=1, secondary_y=False,
                     range=[power_y_min, power_y_max], fixedrange=True)

    # row3 左轴：只标注“绿电消纳量(kWh)”
    fig.update_yaxes(title_text="绿电消纳量(kWh)", row=3, col=1, secondary_y=False,
                     range=[0, left3_y_max], fixedrange=True)
    # row3 右轴：净电价
    price_y_max = max(all_price_max * 1.2, 0.01)   # 动态留 20% 上边距
    fig.update_yaxes(title_text="净电价(元/kWh)", row=3, col=1, secondary_y=True,
                     range=[0, price_y_max], fixedrange=True)

    fig.update_yaxes(title_text="SOC", row=2, col=1,
                     range=[0, 1], fixedrange=True, tickformat=".0%")

    # X轴全局配置，所有子图纵向网格对齐
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E5E5",
        gridwidth=1,
        zeroline=False,
        tickmode="auto",
        nticks=28,
        tickangle=-45,
        tickformat="%m-%d %H:%M",
        hoverformat="%Y-%m-%d %H:%M",
        title_text="",
        minor=dict(
            showgrid=True,
            gridcolor="#F0F0F0",
            gridwidth=0.5,
            # 去掉 dtick，让 Plotly 自动根据视图范围决定次网格密度
            # 全年时约每天一条，单日时每 3~6 小时一条
        )
    )
    return fig
