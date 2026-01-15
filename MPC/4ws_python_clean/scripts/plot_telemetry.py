#!/usr/bin/env python3
"""
基于导出的遥测 CSV 绘制数据图：
- x-y 轨迹坐标图（单独图片）
- 前后轮转角 δf/δr（绘制在同一张图）
- 航向角 ψ（单独图片）
- 横摆率 r（单独图片；若 CSV 无 r，则由 r_dot 数值积分估算）
- β̇（单独图片）

使用方法：
  python3 scripts/plot_telemetry.py [CSV路径]
不传参时默认读取：/Users/gjc/src/4ws/data/telemetry_2025-11-03T07-22-13-599Z.csv
"""

import csv
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def choose_font_for_mac() -> str:
    """选择 macOS 上常见的中文字体，保证中文标题正常显示。"""
    candidates = [
        'PingFang SC',      # macOS 系统中文字体
        'Arial Unicode MS', # 兼容中文的通用字体
        'Heiti SC',         # 黑体（部分系统）
        'Songti SC',        # 宋体（部分系统）
        'Hiragino Sans GB', # 冬青黑体简体中文（部分系统）
        'Noto Sans CJK SC', # Noto CJK
        'Microsoft YaHei',  # 微软雅黑（可能通过第三方安装）
        'DejaVu Sans',      # 兜底，基本符号完整
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return 'DejaVu Sans'


def load_csv_rows(path: str) -> list[dict]:
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def to_float_list(rows: list[dict], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        v = row.get(key)
        if v is None or v == '':
            vals.append(float('nan'))
        else:
            try:
                vals.append(float(v))
            except Exception:
                vals.append(float('nan'))
    return vals


def plot_from_csv(path: str):
    rows = load_csv_rows(path)
    if not rows:
        print(f'CSV 无数据：{path}')
        return

    # 统一时间轴（起点归零，单位秒）
    t_ms = to_float_list(rows, 't_ms')
    t0 = t_ms[0] if t_ms else 0.0
    t_s = [max(0.0, (tm - t0) / 1000.0) if (tm == tm) else float('nan') for tm in t_ms]

    # 读取各列
    x = to_float_list(rows, 'x')
    y = to_float_list(rows, 'y')
    psi = to_float_list(rows, 'psi')
    df = to_float_list(rows, 'df')
    dr = to_float_list(rows, 'dr')
    beta_dot = to_float_list(rows, 'beta_dot')

    # r 列可能不存在：若无，则对 r_dot 做数值积分估算 r
    r: list[float]
    if 'r' in rows[0].keys():
        r = to_float_list(rows, 'r')
    else:
        r_dot = to_float_list(rows, 'r_dot')
        r = [0.0]
        for i in range(1, len(r_dot)):
            dt = (t_ms[i] - t_ms[i - 1]) / 1000.0 if (i < len(t_ms)) else 0.0
            prev = r[-1] if not math.isnan(r[-1]) else 0.0
            inc = r_dot[i - 1] * dt if (not math.isnan(r_dot[i - 1]) and not math.isnan(dt)) else 0.0
            r.append(prev + inc)

    # 字体与负号显示设置
    font_name = choose_font_for_mac()
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False

    # 1) x-y 轨迹
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'b-', linewidth=1.5)
    plt.title('x-y 轨迹', fontsize=14)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.4)

    # 2) 前后轮转角（同图）
    plt.figure(figsize=(7, 4))
    plt.plot(t_s, df, label='δf (rad)', color='tab:blue')
    plt.plot(t_s, dr, label='δr (rad)', color='tab:orange')
    plt.title('前后轮转角', fontsize=14)
    plt.xlabel('t (s)')
    plt.ylabel('角度 (rad)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    # 3) 航向角 ψ
    plt.figure(figsize=(7, 4))
    plt.plot(t_s, psi, color='tab:green')
    plt.title('航向角 ψ', fontsize=14)
    plt.xlabel('t (s)')
    plt.ylabel('ψ (rad)')
    plt.grid(True, linestyle='--', alpha=0.4)

    # 4) 横摆率 r
    plt.figure(figsize=(7, 4))
    plt.plot(t_s, r, color='tab:red')
    plt.title('横摆率 r', fontsize=14)
    plt.xlabel('t (s)')
    plt.ylabel('r (rad/s)')
    plt.grid(True, linestyle='--', alpha=0.4)

    # 5) β̇
    plt.figure(figsize=(7, 4))
    plt.plot(t_s, beta_dot, color='tab:purple')
    plt.title('β̇ ', fontsize=14)
    plt.xlabel('t (s)')
    plt.ylabel('β̇ (rad/s)')
    plt.grid(True, linestyle='--', alpha=0.4)

    # 6) 汇总大图（2×3 排版，轨迹占左侧上方两个位置）
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2)
    ax_xy = fig.add_subplot(gs[0:2, 0])     # 左列上方两格合并
    ax_dfdr = fig.add_subplot(gs[0, 1])     # 右上
    ax_psi = fig.add_subplot(gs[1, 1])      # 右中
    ax_r = fig.add_subplot(gs[2, 0])        # 左下
    ax_beta = fig.add_subplot(gs[2, 1])     # 右下

    # x-y 轨迹（左侧上方两个位置）
    ax_xy.plot(x, y, 'b-', linewidth=1.2)
    ax_xy.set_title('x-y 轨迹', fontsize=12)
    ax_xy.set_xlabel('x (m)')
    ax_xy.set_ylabel('y (m)')
    ax_xy.axis('equal')
    ax_xy.grid(True, linestyle='--', alpha=0.4)

    # 前后轮转角（同图）
    ax_dfdr.plot(t_s, df, label='δf (rad)', color='tab:blue')
    ax_dfdr.plot(t_s, dr, label='δr (rad)', color='tab:orange')
    ax_dfdr.set_title('前后轮转角', fontsize=12)
    ax_dfdr.set_xlabel('t (s)')
    ax_dfdr.set_ylabel('角度 (rad)')
    ax_dfdr.grid(True, linestyle='--', alpha=0.4)
    ax_dfdr.legend()

    # 航向角 ψ（右中）
    ax_psi.plot(t_s, psi, color='tab:green')
    ax_psi.set_title('航向角 ψ', fontsize=12)
    ax_psi.set_xlabel('t (s)')
    ax_psi.set_ylabel('ψ (rad)')
    ax_psi.grid(True, linestyle='--', alpha=0.4)

    # 横摆率 r（左下）
    ax_r.plot(t_s, r, color='tab:red')
    ax_r.set_title('横摆率 r', fontsize=12)
    ax_r.set_xlabel('t (s)')
    ax_r.set_ylabel('r (rad/s)')
    ax_r.grid(True, linestyle='--', alpha=0.4)

    # β̇（右下）
    ax_beta.plot(t_s, beta_dot, color='tab:purple')
    ax_beta.set_title('β̇', fontsize=12)
    ax_beta.set_xlabel('t (s)')
    ax_beta.set_ylabel('β̇ (rad/s)')
    ax_beta.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    # 展示所有图
    plt.show()


def main():
    default_csv = 'data/telemetry_2025-11-14T01-42-18-981Z.csv'
    csv_path = sys.argv[1] if len(sys.argv) >= 2 else default_csv
    if not os.path.exists(csv_path):
        print('文件不存在：', csv_path)
        sys.exit(1)
    plot_from_csv(csv_path)


if __name__ == '__main__':
    main()