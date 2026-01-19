#!/usr/bin/env python3
import csv
import json
import math
import argparse
import os
from typing import List, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def read_config(cfg_path: str):
    a = 1.2; b = 1.6; width = 1.8
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            veh = cfg.get('vehicle', {}) if isinstance(cfg, dict) else {}
            a = float(veh.get('a', a))
            b = float(veh.get('b', b))
            width = float(veh.get('width', width))
    except Exception:
        pass
    return a, b, width

def read_telemetry(csv_path: str) -> List[Dict[str, float]]:
    out = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                out.append({
                    't_ms': float(row.get('t_ms', '0') or 0),
                    'x': float(row.get('x', '0') or 0),
                    'y': float(row.get('y', '0') or 0),
                    'psi': float(row.get('psi', '0') or 0),
                })
            except Exception:
                continue
    return out

def normalize(points: List[Dict[str, float]]):
    if not points:
        return points
    x0 = points[0]['x']; y0 = points[0]['y']; psi0 = points[0]['psi']
    c = math.cos(-psi0); s = math.sin(-psi0)
    for p in points:
        x = p['x'] - x0; y = p['y'] - y0
        xr = x * c - y * s
        yr = x * s + y * c
        p['x'] = xr; p['y'] = yr; p['psi'] = wrap_pi(p['psi'] - psi0)
    return points

def rectangle_corners_center(x: float, y: float, psi: float, L: float, W: float):
    hx = L/2.0; hy = W/2.0
    c = math.cos(psi); s = math.sin(psi)
    def rot(tx, ty):
        return (x + tx*c - ty*s, y + tx*s + ty*c)
    return [
        rot(-hx, -hy),
        rot( hx, -hy),
        rot( hx,  hy),
        rot(-hx,  hy),
        rot(-hx, -hy),
    ]

def plot_envelope(points: List[Dict[str, float]], a: float, b: float, width: float, stride: int):
    L = a + b
    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    # paths
    cx = []; cy = []
    fx = []; fy = []
    rx = []; ry = []
    for i in range(0, len(points), max(1, stride)):
        p = points[i]
        x = p['x']; y = p['y']; psi = p['psi']
        # rectangle (red frame)
        corners = rectangle_corners_center(x, y, psi, L, width)
        xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
        ax.plot(xs, ys, color='red', linewidth=0.8, alpha=0.7)
        # axes centers
        fx.append(x + math.cos(psi) * a); fy.append(y + math.sin(psi) * a)
        rx.append(x - math.cos(psi) * b); ry.append(y - math.sin(psi) * b)
        cx.append(x); cy.append(y)
    # paths overlay with required colors
    ax.plot(fx, fy, color='#ffcc00', linewidth=1.2, alpha=0.95, label='Front axle')
    ax.plot(rx, ry, color='#00a000', linewidth=1.0, alpha=0.9, label='Rear axle')
    ax.plot(cx, cy, color='#1f77b4', linewidth=1.2, alpha=0.95, label='CG path')
    # start/end black frames
    s_corners = rectangle_corners_center(cx[0], cy[0], points[0]['psi'], L, width)
    e_corners = rectangle_corners_center(cx[-1], cy[-1], points[-1]['psi'], L, width)
    ax.plot([c[0] for c in s_corners], [c[1] for c in s_corners], color='black', linewidth=1.4)
    ax.plot([c[0] for c in e_corners], [c[1] for c in e_corners], color='black', linewidth=1.4)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.legend()
    # limits with margin
    try:
        allx = cx + fx + rx; ally = cy + fy + ry
        xmin, xmax = min(allx), max(allx)
        ymin, ymax = min(ally), max(ally)
        mx = (xmax - xmin) * 0.05 + 1e-6
        my = (ymax - ymin) * 0.05 + 1e-6
        ax.set_xlim(xmin - mx, xmax + mx)
        ax.set_ylim(ymin - my, ymax + my)
    except Exception:
        pass
    return fig, ax

def main():
    ap = argparse.ArgumentParser(description='Plot trajectory envelope from telemetry CSV')
    ap.add_argument('--input', required=True, help='telemetry CSV path')
    ap.add_argument('--output', default=None, help='output PNG path')
    ap.add_argument('--config', default='/home/oem/src/4WS/config.json', help='vehicle config json')
    ap.add_argument('--a', type=float, default=None, help='CG to front axle (m)')
    ap.add_argument('--b', type=float, default=None, help='CG to rear axle (m)')
    ap.add_argument('--width', type=float, default=None, help='vehicle width (m)')
    ap.add_argument('--stride', type=int, default=1, help='sample stride for rectangles')
    ap.add_argument('--show', action='store_true', help='show figure before saving')
    ap.add_argument('--normalize', action='store_true', help='normalize start to (0,0,psi=0)')
    args = ap.parse_args()

    a_cfg, b_cfg, w_cfg = read_config(args.config)
    a = args.a if args.a is not None else a_cfg
    b = args.b if args.b is not None else b_cfg
    width = args.width if args.width is not None else w_cfg

    pts = read_telemetry(args.input)
    if not pts or len(pts) < 2:
        raise SystemExit('Not enough telemetry points')
    if args.normalize:
        pts = normalize(pts)
    out_png = args.output or (os.path.splitext(args.input)[0] + '_envelope.png')
    fig, ax = plot_envelope(pts, a=a, b=b, width=width, stride=max(1, args.stride))
    if args.show:
        # switch to interactive backend if available
        try:
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt2  # noqa
        except Exception:
            pass
        plt.show()
    fig.savefig(out_png)
    plt.close(fig)
    print(f'Wrote {out_png}')

if __name__ == '__main__':
    main()
