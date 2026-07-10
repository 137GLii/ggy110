#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空气质量数据分析与可视化脚本
=====================================
数据集: UCI Beijing PM2.5 Data (2010-2014)
功能模块:
  - Module 1: 数据加载与预处理
  - Module 2: 时间序列分析
  - Module 3: 统计指标与相关性
  - Module 4: 多图表可视化
  - Module 5: 季节性变化分析

依赖库: pandas, numpy, matplotlib
"""

import datetime
import io
import os
import urllib.request
import warnings

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器/自动化环境

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
OUTPUT_DIR = '/mnt/agents/output/project'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文字体配置（尝试多个常见字体）
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    'Noto Sans CJK SC', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据集URL
PRIMARY_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'
FALLBACK_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv'

# 输出文件路径
CHART_PATH = os.path.join(OUTPUT_DIR, 'air_quality_analysis.png')
CSV_PATH = os.path.join(OUTPUT_DIR, 'cleaned_air_quality.csv')


def print_section(title):
    """打印带分隔线的章节标题"""
    print('\n' + '=' * 60)
    print(f'  {title}')
    print('=' * 60)


# ============================================================
# Module 1: 数据加载与预处理
# ============================================================
def load_data():
    """
    从网络URL加载空气质量数据，带主/备URL容错机制。
    若网络加载均失败，则生成模拟数据以保证脚本继续运行。
    """
    urls = [PRIMARY_URL, FALLBACK_URL]
    df = None

    for url in urls:
        try:
            print(f'[数据加载] 尝试从 {url} 下载数据...')
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; DataBot/1.0)'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read()
            df = pd.read_csv(io.BytesIO(content))
            print(f'[数据加载] 成功从 {url} 加载数据，共 {len(df)} 行。')
            break
        except Exception as e:
            print(f'[数据加载] 从 {url} 加载失败: {e}')

    if df is None:
        print('[数据加载] 网络加载全部失败，正在生成模拟数据...')
        df = generate_mock_data()

    return df


def generate_mock_data():
    """
    生成模拟的北京PM2.5数据集，结构与真实数据集一致。
    用于网络不可用时保证脚本完整运行。
    """
    print('[模拟数据] 生成 2010-01-01 至 2014-12-31 的每小时模拟数据...')

    # 生成完整的小时级时间索引（5年）
    date_range = pd.date_range(start='2010-01-01', end='2014-12-31 23:00', freq='H')
    n = len(date_range)

    np.random.seed(42)

    # 构造各列
    df = pd.DataFrame({
        'No': range(1, n + 1),
        'year': date_range.year,
        'month': date_range.month,
        'day': date_range.day,
        'hour': date_range.hour,
        'pm2.5': np.round(np.random.normal(loc=95, scale=60, size=n).clip(10, 500), 1),
        'DEWP': np.round(np.random.normal(loc=-5, scale=15, size=n), 1),
        'TEMP': np.round(np.random.normal(loc=12, scale=12, size=n), 1),
        'PRES': np.round(np.random.normal(loc=1015, scale=10, size=n), 1),
        'cbwd': np.random.choice(['NE', 'NW', 'SE', 'cv'], size=n),
        'Iws': np.round(np.random.exponential(scale=15, size=n), 1),
        'Is': np.round(np.random.exponential(scale=0.5, size=n), 2),
        'Ir': np.round(np.random.exponential(scale=0.8, size=n), 2),
    })

    # 人为制造一些缺失值以模拟真实场景
    missing_idx = np.random.choice(df.index, size=int(n * 0.03), replace=False)
    df.loc[missing_idx, 'pm2.5'] = np.nan

    print(f'[模拟数据] 生成完成，共 {len(df)} 行（{n // 24} 天）。')
    return df


def preprocess_data(df):
    """
    数据预处理流程：
      1. 合并 year/month/day/hour 为 datetime 列
      2. PM2.5 缺失值线性插值填充
      3. 创建时间衍生特征: year, month, day, hour, season, weekday
    """
    print('[预处理] 开始数据清洗与特征工程...')

    df = df.copy()

    # 统一列名（备用数据集的列名可能不同）
    col_mapping = {
        'pollution': 'pm2.5',
        'wnd_spd': 'Iws',
        'snow': 'Is',
        'rain': 'Ir',
    }
    df.rename(columns=col_mapping, inplace=True)

    # 确保核心列存在
    for col in ['year', 'month', 'day', 'hour']:
        if col not in df.columns:
            df[col] = 0

    # 构造 datetime 列
    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour']],
        errors='coerce'
    )

    # 若有解析失败的日期，尝试从索引推断
    if df['datetime'].isna().sum() > 0:
        fallback_dates = pd.date_range(
            start='2010-01-01', periods=len(df), freq='H'
        )
        df['datetime'] = df['datetime'].fillna(
            pd.Series(fallback_dates, index=df.index)
        )

    # 设置 datetime 为索引
    df.set_index('datetime', inplace=True)

    # PM2.5 缺失值检测与线性插值
    pm25_missing_before = df['pm2.5'].isna().sum()
    df['pm2.5'] = df['pm2.5'].interpolate(method='linear')
    # 若首尾仍有缺失，使用前向/后向填充
    df['pm2.5'] = df['pm2.5'].ffill().bfill()
    pm25_missing_after = df['pm2.5'].isna().sum()
    print(f'[预处理] PM2.5 缺失值: 处理前 {pm25_missing_before} 个, 处理后 {pm25_missing_after} 个')

    # 创建时间衍生特征
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday  # 0=周一, 6=周日

    # 季节定义: 春(3-5), 夏(6-8), 秋(9-11), 冬(12-2)
    def get_season(month):
        if month in [3, 4, 5]:
            return '春'
        elif month in [6, 7, 8]:
            return '夏'
        elif month in [9, 10, 11]:
            return '秋'
        else:
            return '冬'

    df['season'] = df['month'].apply(get_season)

    print(f'[预处理] 完成。数据形状: {df.shape}，时间范围: {df.index.min()} ~ {df.index.max()}')
    return df


# ============================================================
# Module 2: 时间序列分析
# ============================================================
def time_series_analysis(df):
    """
    生成时间序列统计数据：
      - PM2.5 日平均值
      - PM2.5 月平均值
      - 24小时平均浓度变化（日内模式）
    """
    print('[时间序列] 计算时间序列聚合数据...')

    # 日平均值
    daily_avg = df['pm2.5'].resample('D').mean()

    # 月平均值
    monthly_avg = df['pm2.5'].resample('ME').mean()

    # 24小时日内模式
    hourly_pattern = df.groupby('hour')['pm2.5'].mean()

    print(f'[时间序列] 日平均: {len(daily_avg)} 天, 月平均: {len(monthly_avg)} 月, 24小时模式已计算')
    return daily_avg, monthly_avg, hourly_pattern


# ============================================================
# Module 3: 统计指标与相关性
# ============================================================
def compute_statistics(df):
    """
    计算数值列的统计描述和皮尔逊相关系数矩阵。
    """
    print('[统计指标] 计算描述性统计和相关性矩阵...')

    # 数值列
    numeric_cols = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    available_cols = [c for c in numeric_cols if c in df.columns]

    # 描述性统计
    desc = df[available_cols].describe()
    print('\n[描述性统计]')
    print(desc.round(2))

    # 皮尔逊相关系数矩阵
    corr_matrix = df[available_cols].corr(method='pearson')
    print('\n[皮尔逊相关系数矩阵]')
    print(corr_matrix.round(3))

    return desc, corr_matrix


# ============================================================
# Module 4: 多图表可视化 (Matplotlib 纯实现)
# ============================================================
def draw_correlation_heatmap(ax, corr_matrix):
    """
    使用 Matplotlib 绘制相关系数矩阵热力图（不使用 seaborn）。
    """
    data = corr_matrix.values
    n = len(corr_matrix)
    labels = corr_matrix.columns.tolist()

    # 颜色映射: 正相关=暖色, 负相关=冷色
    cmap = plt.cm.RdYlGn  # 红-黄-绿
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # 设置刻度
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # 在每个格子中显示数值
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    ax.set_title('相关系数矩阵热力图', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_month_hour_heatmap(ax, df):
    """
    绘制 月份×小时 的 PM2.5 平均浓度热力图。
    """
    pivot = df.pivot_table(
        values='pm2.5', index='month', columns='hour', aggfunc='mean'
    )
    data = pivot.values

    cmap = plt.cm.YlOrRd  # 黄-橙-红
    im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest')

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2), fontsize=8)
    ax.set_yticks(range(12))
    ax.set_yticklabels([f'{i+1}月' for i in range(12)], fontsize=9)
    ax.set_xlabel('小时', fontsize=10)
    ax.set_ylabel('月份', fontsize=10)
    ax.set_title('月份×小时 PM2.5 平均浓度热力图', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='PM2.5 (ug/m^3)')


def create_visualization(df, daily_avg, monthly_avg, hourly_pattern, corr_matrix):
    """
    创建 2×4 子图布局的综合可视化大图，尺寸 20×16，DPI 150。
    """
    print('[可视化] 开始生成 2x4 综合图表...')

    fig, axes = plt.subplots(2, 4, figsize=(20, 16), dpi=150)
    fig.suptitle('北京空气质量数据分析综合报告 (2010-2014)', fontsize=16, fontweight='bold')

    # ---------- 子图1: PM2.5 日均浓度时间序列折线图 ----------
    ax1 = axes[0, 0]
    ax1.plot(daily_avg.index, daily_avg.values, color='steelblue', linewidth=0.6, alpha=0.8)
    ax1.set_title('PM2.5 日均浓度时间序列', fontsize=12)
    ax1.set_xlabel('日期', fontsize=10)
    ax1.set_ylabel('PM2.5 (ug/m^3)', fontsize=10)
    ax1.tick_params(axis='x', rotation=30, labelsize=8)
    ax1.grid(True, alpha=0.3)

    # ---------- 子图2: 各月平均 PM2.5 浓度柱状图 ----------
    ax2 = axes[0, 1]
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                   '7月', '8月', '9月', '10月', '11月', '12月']
    month_means = df.groupby('month')['pm2.5'].mean()
    bars = ax2.bar(range(1, 13), month_means.values, color='coral', edgecolor='white')
    ax2.set_title('各月平均 PM2.5 浓度', fontsize=12)
    ax2.set_xlabel('月份', fontsize=10)
    ax2.set_ylabel('PM2.5 (ug/m^3)', fontsize=10)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bar, val in zip(bars, month_means.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{val:.0f}', ha='center', va='bottom', fontsize=7)

    # ---------- 子图3: 各风向的平均 PM2.5 柱状图 ----------
    ax3 = axes[0, 2]
    if 'cbwd' in df.columns:
        wind_means = df.groupby('cbwd')['pm2.5'].mean().sort_values(ascending=False)
        bars3 = ax3.bar(wind_means.index.astype(str), wind_means.values, color='teal', edgecolor='white')
        ax3.set_title('各风向平均 PM2.5 浓度', fontsize=12)
        ax3.set_xlabel('风向', fontsize=10)
        ax3.set_ylabel('PM2.5 (ug/m^3)', fontsize=10)
        ax3.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars3, wind_means.values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    # ---------- 子图4: PM2.5 vs 温度 散点图 ----------
    ax4 = axes[0, 3]
    if 'TEMP' in df.columns:
        # 采样以避免散点过多
        sample_df = df.sample(min(5000, len(df)), random_state=42)
        ax4.scatter(sample_df['TEMP'], sample_df['pm2.5'], alpha=0.3, s=8, color='purple')
        ax4.set_title('PM2.5 vs 温度', fontsize=12)
        ax4.set_xlabel('温度 (TEMP, °C)', fontsize=10)
        ax4.set_ylabel('PM2.5 (ug/m^3)', fontsize=10)
        ax4.grid(True, alpha=0.3)

    # ---------- 子图5: PM2.5 vs 风速 散点图 ----------
    ax5 = axes[1, 0]
    if 'Iws' in df.columns:
        sample_df = df.sample(min(5000, len(df)), random_state=43)
        ax5.scatter(sample_df['Iws'], sample_df['pm2.5'], alpha=0.3, s=8, color='green')
        ax5.set_title('PM2.5 vs 风速', fontsize=12)
        ax5.set_xlabel('风速 (Iws, m/s)', fontsize=10)
        ax5.set_ylabel('PM2.5 (ug/m^3)', fontsize=10)
        ax5.grid(True, alpha=0.3)

    # ---------- 子图6: 24小时日内模式折线图 ----------
    ax6 = axes[1, 1]
    ax6.plot(hourly_pattern.index, hourly_pattern.values,
             marker='o', color='darkorange', linewidth=2, markersize=5)
    ax6.set_title('24小时平均 PM2.5 浓度变化', fontsize=12)
    ax6.set_xlabel('小时', fontsize=10)
    ax6.set_ylabel('PM2.5 (ug/m^3)', fontsize=10)
    ax6.set_xticks(range(0, 24, 2))
    ax6.grid(True, alpha=0.3)

    # ---------- 子图7: 相关系数矩阵热力图 ----------
    ax7 = axes[1, 2]
    draw_correlation_heatmap(ax7, corr_matrix)

    # ---------- 子图8: 月份×小时 PM2.5 热力图 ----------
    ax8 = axes[1, 3]
    draw_month_hour_heatmap(ax8, df)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[可视化] 综合图表已保存到: {CHART_PATH}')


# ============================================================
# Module 5: 季节性变化分析
# ============================================================
def seasonal_analysis(df):
    """
    分析各季节 PM2.5 的统计指标（均值、中位数、标准差、最大值），
    并绘制季节性柱状图。
    """
    print_section('Module 5: 季节性变化分析')

    season_order = ['春', '夏', '秋', '冬']
    season_stats = []

    for s in season_order:
        subset = df[df['season'] == s]['pm2.5']
        stats = {
            '季节': s,
            '均值': round(subset.mean(), 2),
            '中位数': round(subset.median(), 2),
            '标准差': round(subset.std(), 2),
            '最大值': round(subset.max(), 2),
            '数据量': len(subset),
        }
        season_stats.append(stats)

    season_df = pd.DataFrame(season_stats)
    print('\n[季节统计指标]')
    print(season_df.to_string(index=False))

    # 绘制季节性柱状图（四季对比）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    fig.suptitle('PM2.5 季节性变化分析', fontsize=14, fontweight='bold')

    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']  # 绿、蓝、橙、红

    # 均值
    ax1 = axes[0]
    bars1 = ax1.bar(season_df['季节'], season_df['均值'], color=colors, edgecolor='white')
    ax1.set_title('各季节 PM2.5 平均浓度', fontsize=12)
    ax1.set_ylabel('PM2.5 (ug/m^3)', fontsize=10)
    for bar, val in zip(bars1, season_df['均值']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)

    # 中位数
    ax2 = axes[1]
    bars2 = ax2.bar(season_df['季节'], season_df['中位数'], color=colors, edgecolor='white')
    ax2.set_title('各季节 PM2.5 中位浓度', fontsize=12)
    ax2.set_ylabel('PM2.5 (ug/m^3)', fontsize=10)
    for bar, val in zip(bars2, season_df['中位数']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)

    # 标准差
    ax3 = axes[2]
    bars3 = ax3.bar(season_df['季节'], season_df['标准差'], color=colors, edgecolor='white')
    ax3.set_title('各季节 PM2.5 浓度波动（标准差）', fontsize=12)
    ax3.set_ylabel('PM2.5 (ug/m^3)', fontsize=10)
    for bar, val in zip(bars3, season_df['标准差']):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    ax3.grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # 保存季节性图表
    seasonal_chart_path = os.path.join(OUTPUT_DIR, 'seasonal_analysis.png')
    plt.savefig(seasonal_chart_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[季节性图表] 已保存到: {seasonal_chart_path}')

    # 输出季节性结论文字
    print('\n[季节性分析结论]')
    mean_values = season_df.set_index('季节')['均值']
    highest_season = mean_values.idxmax()
    lowest_season = mean_values.idxmin()

    print(f'  1. PM2.5 浓度最高的季节是「{highest_season}季」，平均浓度为 {mean_values[highest_season]:.2f} ug/m^3')
    print(f'  2. PM2.5 浓度最低的季节是「{lowest_season}季」，平均浓度为 {mean_values[lowest_season]:.2f} ug/m^3')
    print(f'  3. 季节间差异: {mean_values.max() - mean_values.min():.2f} ug/m^3')

    # 冬季 vs 夏季倍数
    if '冬' in mean_values and '夏' in mean_values:
        ratio = mean_values['冬'] / max(mean_values['夏'], 0.01)
        print(f'  4. 冬季平均浓度约为夏季的 {ratio:.2f} 倍')

    # 给出定性结论
    std_vals = season_df.set_index('季节')['标准差']
    most_variable = std_vals.idxmax()
    print(f'  5. 污染波动最大的季节是「{most_variable}季」（标准差 {std_vals[most_variable]:.2f}）')

    return season_df


# ============================================================
# 主流程
# ============================================================
def main():
    """主入口函数：按模块顺序执行全部分析流程"""
    start_time = datetime.datetime.now()
    print_section('北京空气质量数据分析脚本启动')
    print(f'开始时间: {start_time}')

    # ---------- Module 1: 数据加载与预处理 ----------
    print_section('Module 1: 数据加载与预处理')
    raw_df = load_data()
    df = preprocess_data(raw_df)

    print('\n[数据概览]')
    print(f'数据行数: {len(df)}')
    print(f'数据列数: {len(df.columns)}')
    print(f'列名: {list(df.columns)}')
    print(f'时间范围: {df.index.min()} ~ {df.index.max()}')
    print('\n[前5行数据预览]')
    print(df.head())
    print('\n[数据类型]')
    print(df.dtypes)

    # ---------- Module 2: 时间序列分析 ----------
    print_section('Module 2: 时间序列分析')
    daily_avg, monthly_avg, hourly_pattern = time_series_analysis(df)
    print(f'日均浓度范围: {daily_avg.min():.2f} ~ {daily_avg.max():.2f}')
    print(f'月均值范围: {monthly_avg.min():.2f} ~ {monthly_avg.max():.2f}')
    print(f'24小时最低均值(小时 {hourly_pattern.idxmin()}): {hourly_pattern.min():.2f}')
    print(f'24小时最高均值(小时 {hourly_pattern.idxmax()}): {hourly_pattern.max():.2f}')

    # ---------- Module 3: 统计指标与相关性 ----------
    print_section('Module 3: 统计指标与相关性')
    desc, corr_matrix = compute_statistics(df)

    # 输出 PM2.5 与气象因素的关键相关性
    if 'TEMP' in corr_matrix.columns:
        print(f'\nPM2.5 与温度的相关系数: {corr_matrix.loc["pm2.5", "TEMP"]:.3f}')
    if 'PRES' in corr_matrix.columns:
        print(f'PM2.5 与气压的相关系数: {corr_matrix.loc["pm2.5", "PRES"]:.3f}')
    if 'Iws' in corr_matrix.columns:
        print(f'PM2.5 与风速的相关系数: {corr_matrix.loc["pm2.5", "Iws"]:.3f}')

    # ---------- Module 4: 多图表可视化 ----------
    print_section('Module 4: 多图表可视化')
    create_visualization(df, daily_avg, monthly_avg, hourly_pattern, corr_matrix)

    # ---------- Module 5: 季节性变化分析 ----------
    season_df = seasonal_analysis(df)

    # ---------- 保存清洗后数据 ----------
    df.to_csv(CSV_PATH, encoding='utf-8-sig')
    print(f'\n[数据保存] 清洗后数据已保存到: {CSV_PATH}')
    print(f'CSV文件大小: {os.path.getsize(CSV_PATH) / 1024:.1f} KB')

    # ---------- 完成总结 ----------
    end_time = datetime.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print_section('分析完成')
    print(f'结束时间: {end_time}')
    print(f'总耗时: {elapsed:.2f} 秒')
    print(f'生成文件:')
    print(f'  - {CHART_PATH}')
    print(f'  - {CSV_PATH}')
    seasonal_chart = os.path.join(OUTPUT_DIR, 'seasonal_analysis.png')
    if os.path.exists(seasonal_chart):
        print(f'  - {seasonal_chart}')

    return df, daily_avg, monthly_avg, hourly_pattern, corr_matrix, season_df


# ============================================================
# 脚本入口
# ============================================================
if __name__ == '__main__':
    main()
