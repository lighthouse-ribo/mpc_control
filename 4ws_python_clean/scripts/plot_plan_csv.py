#!/usr/bin/env python3
import csv
import math
import argparse
import os

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def read_plan(csv_path: str):
    pts = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None or [h.strip() for h in rd.fieldnames] != ['t','x','y','psi']:
            raise RuntimeError('CSV 表头必须为: t,x,y,psi')
        for row in rd:
            try:
                t = float(row.get('t', '0') or 0)
                x = float(row.get('x'))
                y = float(row.get('y'))
                psi = float(row.get('psi', '0') or 0)
            except Exception:
                continue
            pts.append({'t': t, 'x': x, 'y': y, 'psi': psi})
    if len(pts) < 2:
        raise RuntimeError('点数不足')
    return pts

def normalize(pts):
    x0 = pts[0]['x']; y0 = pts[0]['y']; psi0 = pts[0]['psi']
    c = math.cos(-psi0); s = math.sin(-psi0)
    out = []
    for p in pts:
        x = p['x'] - x0; y = p['y'] - y0
        xr = x * c - y * s
        yr = x * s + y * c
        out.append({'t': p['t'], 'x': xr, 'y': yr, 'psi': wrap_pi(p['psi'] - psi0)})
    return out

def plot_plan(pts, stride=1, arrows_every=100, out_png=None,
              figw=14, figh=10, dpi=180, lw=2.0,
              arrow_scale=1.2, arrow_width=0.004, pad_frac=0.10):
    xs = [p['x'] for p in pts[::stride]]
    ys = [p['y'] for p in pts[::stride]]
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=dpi)
    ax.plot(xs, ys, color='#1f77b4', linewidth=lw)
    # 起止方框
    def draw_box(x, y, psi, L=0.5, W=0.25, color='black'):
        hx = L/2; hy = W/2; c = math.cos(psi); s = math.sin(psi)
        def rot(tx, ty): return (x + tx*c - ty*s, y + tx*s + ty*c)
        corners = [rot(-hx,-hy), rot(hx,-hy), rot(hx,hy), rot(-hx,hy), rot(-hx,-hy)]
        ax.plot([c[0] for c in corners], [c[1] for c in corners], color=color, linewidth=1.2)
    draw_box(pts[0]['x'], pts[0]['y'], pts[0]['psi'], color='black')
    draw_box(pts[-1]['x'], pts[-1]['y'], pts[-1]['psi'], color='black')
    # 航向箭头
    qx = []; qy = []; u = []; v = []
    step = max(1, arrows_every)
    for i in range(0, len(pts), step):
        qx.append(pts[i]['x']); qy.append(pts[i]['y'])
        u.append(math.cos(pts[i]['psi']))
        v.append(math.sin(pts[i]['psi']))
    ax.quiver(qx, qy, u, v, color='#ffcc00', scale_units='xy', scale=arrow_scale, width=arrow_width)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title('Plan Trajectory (t,x,y,psi)')
    # 视图边距：按数据范围加 pad
    try:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        mx = (xmax - xmin) * pad_frac + 1e-6
        my = (ymax - ymin) * pad_frac + 1e-6
        ax.set_xlim(xmin - mx, xmax + mx)
        ax.set_ylim(ymin - my, ymax + my)
    except Exception:
        pass
    if out_png:
        plt.savefig(out_png, dpi=160)
    plt.show()

def main():
    ap = argparse.ArgumentParser(description='Plot plan CSV (t,x,y,psi) with Matplotlib')
    ap.add_argument('--input', default='/home/oem/src/4WS/data/Trajectory/plan_turnleft_curve4_merged copy.csv')
    ap.add_argument('--output', default=None)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--normalize', action='store_true')
    ap.add_argument('--figw', type=float, default=14)
    ap.add_argument('--figh', type=float, default=10)
    ap.add_argument('--dpi', type=int, default=180)
    ap.add_argument('--lw', type=float, default=2.0)
    ap.add_argument('--arrows-every', type=int, default=50)
    ap.add_argument('--arrow-scale', type=float, default=1.2)
    ap.add_argument('--arrow-width', type=float, default=0.004)
    ap.add_argument('--pad-frac', type=float, default=0.10)
    args = ap.parse_args()
    pts = read_plan(args.input)
    if args.normalize:
        pts = normalize(pts)
    plot_plan(
        pts,
        stride=max(1, args.stride),
        arrows_every=max(1, args.arrows_every),
        out_png=args.output,
        figw=float(args.figw), figh=float(args.figh), dpi=int(args.dpi), lw=float(args.lw),
        arrow_scale=float(args.arrow_scale), arrow_width=float(args.arrow_width), pad_frac=float(args.pad_frac),
    )

if __name__ == '__main__':
    main()
