#!/usr/bin/env python3
import csv
import math
import argparse
import os
import glob

R = 6378137.0

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def convert_group(rows, heading_ref: str = 'east', dt: float = 0.1):
    has_order = all(('order' in r and str(r['order']).strip() != '') for r in rows)
    if has_order:
        try:
            rows = sorted(rows, key=lambda r: int(r['order']))
        except Exception:
            pass
    lon0 = float(rows[0]['lon'])
    lat0 = float(rows[0]['lat'])
    lat0_rad = math.radians(lat0)
    kx = (math.pi / 180.0) * R * math.cos(lat0_rad)
    ky = (math.pi / 180.0) * R
    out = []
    for i, r in enumerate(rows):
        lon = float(r['lon']); lat = float(r['lat']); hdg = float(r['heading'])
        x = (lon - lon0) * kx
        y = (lat - lat0) * ky
        psi = hdg if heading_ref == 'east' else (math.pi / 2.0 - hdg)
        psi = wrap_pi(psi)
        t = i * dt
        out.append({'t': t, 'x': x, 'y': y, 'psi': psi})
    return out

def rotate_normalize(points):
    if not points:
        return points
    psi0 = float(points[0].get('psi', 0.0))
    c = math.cos(-psi0)
    s = math.sin(-psi0)
    out = []
    for p in points:
        x = float(p.get('x', 0.0))
        y = float(p.get('y', 0.0))
        xr = x * c - y * s
        yr = x * s + y * c
        psi = wrap_pi(float(p.get('psi', 0.0)) - psi0)
        out.append({'t': float(p.get('t', 0.0)), 'x': xr, 'y': yr, 'psi': psi})
    return out

def clean_and_smooth(points, dup_eps=1e-3, max_step=0.5, smooth_window=5, dt=0.1,
                     curve_step=0.05, kappa_threshold=0.02):
    if not points or len(points) < 2:
        return points
    cleaned = []
    for i, p in enumerate(points):
        if i == 0:
            cleaned.append(p)
            continue
        dx = p['x'] - cleaned[-1]['x']
        dy = p['y'] - cleaned[-1]['y']
        ds = math.hypot(dx, dy)
        if ds <= dup_eps:
            continue
        cleaned.append(p)
    if len(cleaned) < 2:
        return cleaned
    densified = []
    prev_ang = None
    for i in range(len(cleaned) - 1):
        p0 = cleaned[i]
        p1 = cleaned[i + 1]
        densified.append(p0)
        dx = p1['x'] - p0['x']
        dy = p1['y'] - p0['y']
        ds = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        if prev_ang is None:
            prev_ang = ang
        dpsi = (ang - prev_ang + math.pi) % (2*math.pi) - math.pi
        prev_ang = ang
        # 计算本段使用的目标步长：弯道更密集
        step_target = max_step
        kappa = abs(dpsi) / max(1e-6, ds)
        # 自适应步长：曲率越大步长越小
        gain = 30.0
        step_target = min(step_target, max(1e-3, step_target / (1.0 + gain * kappa)))
        if kappa >= kappa_threshold:
            step_target = min(step_target, curve_step)
            # 弯道最小密度：每米至少 ppm_curve 个点
            ppm_curve = 10.0
            k_min = int(math.ceil(ds * ppm_curve))
            if ds > step_target:
                k = max(k_min, int(math.ceil(ds / step_target)))
            else:
                k = k_min
        else:
            k = int(math.ceil(ds / step_target)) if ds > step_target else 0
        if k and k > 0:
            for j in range(1, k):
                r = j / k
                xi = p0['x'] + r * dx
                yi = p0['y'] + r * dy
                densified.append({'t': 0.0, 'x': xi, 'y': yi, 'psi': 0.0})
    densified.append(cleaned[-1])
    m = len(densified)
    # recompute psi by geometry
    for i in range(m):
        if i < m - 1:
            dx = densified[i + 1]['x'] - densified[i]['x']
            dy = densified[i + 1]['y'] - densified[i]['y']
        else:
            dx = densified[i]['x'] - densified[i - 1]['x']
            dy = densified[i]['y'] - densified[i - 1]['y']
        densified[i]['psi'] = wrap_pi(math.atan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-9 else (densified[i-1]['psi'] if i>0 else 0.0)
    # smooth psi with circular mean
    smooth_window = max(3, smooth_window | 1)
    half = smooth_window // 2
    psi_sm = []
    for i in range(m):
        a = max(0, i - half)
        b = min(m - 1, i + half)
        ss = 0.0; cc = 0.0
        for k in range(a, b + 1):
            ss += math.sin(densified[k]['psi'])
            cc += math.cos(densified[k]['psi'])
        psi_sm.append(wrap_pi(math.atan2(ss, cc)))
    for i in range(m):
        densified[i]['psi'] = psi_sm[i]
        densified[i]['t'] = i * dt
    return densified

def resample_by_arclength(points, ds_straight=0.02, ds_curve=0.01, kappa_threshold=0.02):
    if not points or len(points) < 2:
        return points
    # 计算每段长度与方向角
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    ps = []
    ds_list = []
    for i in range(len(points) - 1):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        ds = math.hypot(dx, dy)
        ds_list.append(ds)
        ang = math.atan2(dy, dx) if ds > 1e-9 else points[i].get('psi', 0.0)
        ps.append(ang)
    ps.append(ps[-1] if ps else 0.0)
    # 局部曲率（用相邻段方向差分）
    kappa_loc = [0.0] * (len(points) - 1)
    for i in range(len(points) - 1):
        j_prev = max(0, i - 1)
        j_next = min(len(points) - 2, i + 1)
        dpsi = (ps[j_next] - ps[j_prev] + math.pi) % (2*math.pi) - math.pi
        ds_avg = max(1e-6, 0.5 * (ds_list[j_prev] + ds_list[j_next]))
        kappa_loc[i] = abs(dpsi) / ds_avg
    # 累积弧长
    S = [0.0]
    for ds in ds_list:
        S.append(S[-1] + ds)
    Ltot = S[-1]
    # 构造目标采样位置
    out_pts = []
    s = 0.0
    idx = 0
    while s < Ltot:
        while idx < len(S)-1 and S[idx+1] < s:
            idx += 1
        # 局部步长选择
        step = ds_straight
        if idx < len(kappa_loc) and kappa_loc[idx] >= kappa_threshold:
            step = ds_curve
        # 线性插值 x,y
        if idx >= len(S)-1:
            out_pts.append({'x': xs[-1], 'y': ys[-1], 'psi': ps[-1]})
            break
        s0 = S[idx]; s1 = S[idx+1]
        r = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        x = xs[idx] + r * (xs[idx+1] - xs[idx])
        y = ys[idx] + r * (ys[idx+1] - ys[idx])
        # 航向用邻点方向重算
        ang = math.atan2(ys[idx+1] - ys[idx], xs[idx+1] - xs[idx]) if (s1 - s0) > 1e-9 else ps[idx]
        out_pts.append({'x': x, 'y': y, 'psi': ang})
        s += step
    # 圆均值平滑航向并重建 t
    w = 5
    half = w // 2
    psi_sm = []
    for i in range(len(out_pts)):
        a = max(0, i - half)
        b = min(len(out_pts) - 1, i + half)
        ss = 0.0; cc = 0.0
        for k in range(a, b + 1):
            ss += math.sin(out_pts[k]['psi'])
            cc += math.cos(out_pts[k]['psi'])
        psi_sm.append((math.atan2(ss, cc)))
    for i in range(len(out_pts)):
        out_pts[i]['psi'] = psi_sm[i]
        out_pts[i]['t'] = i * (0.05)  # 给一个较小的时间步标注，可被下游忽略
    return out_pts

def main():
    ap = argparse.ArgumentParser(description='Convert lon,lat,heading CSV to plan t,x,y,psi CSV')
    ap.add_argument('--input', default='/home/oem/src/4WS/documents/pts_yangluo_follow.csv')
    ap.add_argument('--input-dir', default=None, help='directory containing CSV files to batch convert')
    ap.add_argument('--glob', default='*.csv', help='glob pattern for batch input')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--heading-ref', choices=['east','north'], default='east', help='heading reference: east or north')
    ap.add_argument('--dt', type=float, default=0.1, help='time step seconds')
    ap.add_argument('--dup-eps', type=float, default=1e-3, help='duplicate removal threshold (m)')
    ap.add_argument('--max-step', type=float, default=1.0, help='max segment length before densify (m)')
    ap.add_argument('--smooth-window', type=int, default=5, help='psi smoothing window size (odd)')
    ap.add_argument('--no-densify', action='store_true', help='disable densify/resample steps')
    ap.add_argument('--normalize', action='store_true', help='normalize start to (0,0,psi=0)')
    args = ap.parse_args()

    def process_file(path: str):
        outdir_local = args.outdir or os.path.dirname(path) or '.'
        rows = []
        with open(path, 'r', newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for row in rd:
                try:
                    lon = float(row.get('lon'))
                    lat = float(row.get('lat'))
                    hdg = float(row.get('heading'))
                except Exception:
                    continue
                rec = {'lon': lon, 'lat': lat, 'heading': hdg}
                if 'order' in row:
                    rec['order'] = row.get('order')
                rows.append(rec)
        if not rows:
            print(f'No valid rows in {path}')
            return
        os.makedirs(outdir_local, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        pts = convert_group(rows, heading_ref=args.heading_ref, dt=args.dt)
        if args.normalize:
            pts = rotate_normalize(pts)
        if not args.no_densify:
            pts = clean_and_smooth(pts, dup_eps=args.dup_eps, max_step=args.max_step, smooth_window=args.smooth_window, dt=args.dt,
                                    curve_step=0.05, kappa_threshold=0.02)
            pts = resample_by_arclength(pts, ds_straight=0.02, ds_curve=0.01, kappa_threshold=0.02)
        out_path = os.path.join(outdir_local, f'plan11_{stem}.csv')
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['t','x','y','psi'])
            for p in pts:
                wr.writerow([f"{p['t']:.3f}", f"{p['x']:.6f}", f"{p['y']:.6f}", f"{p['psi']:.9f}"])
        print(f'Wrote {out_path} ({len(pts)} points)')

    if args.input_dir:
        pattern = os.path.join(args.input_dir, args.glob)
        files = sorted(glob.glob(pattern))
        if not files:
            print(f'No files matched: {pattern}')
        for fp in files:
            process_file(fp)
    else:
        process_file(args.input)

if __name__ == '__main__':
    main()
