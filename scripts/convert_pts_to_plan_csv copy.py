#!/usr/bin/env python3
import csv
import math
import argparse
import os
import glob
import numpy as np
from scipy.interpolate import splprep, splev

R = 6378137.0

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def convert_group(rows, heading_ref: str = 'east', dt: float = 0.1):
    
    # --- 【V4 修正】修复数据排序 ---
    # 检查 'order' 列是否存在且有效，并据此排序
    if rows and 'order' in rows[0]:
        print("Info: 检测到 'order' 列。正在尝试排序...")
        sorted_rows_with_key = []
        invalid_rows = []
        
        for r in rows:
            order_val = str(r.get('order', '')).strip()
            try:
                # 尝试将 'order' 转换为整数
                order_num = int(order_val)
                # 添加 (排序键, 数据) 元组
                sorted_rows_with_key.append((order_num, r))
            except (ValueError, TypeError):
                # 如果 'order' 是空的或无效的 (非整数)，则丢弃
                invalid_rows.append(r)
        
        if sorted_rows_with_key:
            # 1. 按照 order_num (元组的第一个元素) 排序
            sorted_rows_with_key.sort(key=lambda x: x[0])
            # 2. 提取排序后的行数据
            rows = [r for order_num, r in sorted_rows_with_key]
            print(f"Info: 成功按 'order' 排序 {len(rows)} 个点。")
            if invalid_rows:
                print(f"Warning: 跳过了 {len(invalid_rows)} 个 'order' 值无效或缺失的点。")
        else:
            # 'order' 列存在，但没有一个是有效的整数
            print("Warning: 'order' 列存在，但没有找到有效的整数值。将使用文件原始顺序。")
            # 'rows' 保持原样 (未排序)
    else:
        # 'order' 列不存在
        print("Info: 未找到 'order' 列。将使用文件原始顺序。")
    # --- 排序逻辑结束 ---

    if not rows: # 如果排序后没有点/或原始即为空
        return []

    # 按（现在已正确排序的）第一个点设置坐标系原点
    lon0 = float(rows[0]['lon'])
    lat0 = float(rows[0]['lat'])
    lat0_rad = math.radians(lat0)
    kx = (math.pi / 180.0) * R * math.cos(lat0_rad)
    ky = (math.pi / 180.0) * R
    out = []
    
    # 遍历已正确排序的行
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


# (V3 函数 - 保留) 这是正确的平滑样条函数
def resample_with_smoothing_spline(points, dup_eps=1e-3, smooth_factor=1.0, 
                                   num_factor=20, dt=0.1, smooth_window=5):
    """
    使用 B 样条 *平滑* (splprep)。
    只要输入的 'points' 列表是按顺序排列的，这个函数就是正确的。
    """
    if not points or len(points) < 2:
        return points
    
    # 1. 清理重复点
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
    
    m = len(cleaned)
    if m < 4: # B样条 (k=3) 至少需要 4 个点
        print(f"Warning: 点数 ({m}) < 4，无法进行B样条拟合。")
        for i, p in enumerate(cleaned):
            p['t'] = i * dt
        return cleaned

    # 2. 提取 x, y 坐标
    xs = np.array([p['x'] for p in cleaned])
    ys = np.array([p['y'] for p in cleaned])

    # 3. 拟合平滑样条
    try:
        tck, u = splprep([xs, ys], s=smooth_factor, k=3, per=False)
    except Exception as e:
        print(f"B样条拟合失败: {e}. (数据点是否仍有乱序?)")
        return cleaned

    # 4. 沿新的平滑曲线进行密集重采样
    num_new = m * num_factor
    u_new = np.linspace(u.min(), u.max(), num_new)
    
    x_new, y_new = splev(u_new, tck)
    dx_du, dy_du = splev(u_new, tck, der=1)

    out_pts = []
    for i in range(num_new):
        x = x_new[i]
        y = y_new[i]
        psi = wrap_pi(math.atan2(dy_du[i], dx_du[i]))
        out_pts.append({'t': 0.0, 'x': x, 'y': y, 'psi': psi})

    # 5. （可选）对导出的 psi 进行圆周均值平滑
    m = len(out_pts)
    smooth_window = max(3, smooth_window | 1)
    half = smooth_window // 2
    psi_sm = []
    for i in range(m):
        a = max(0, i - half)
        b = min(m - 1, i + half)
        ss = 0.0; cc = 0.0
        for k in range(a, b + 1):
            ss += math.sin(out_pts[k]['psi'])
            cc += math.cos(out_pts[k]['psi'])
        psi_sm.append(wrap_pi(math.atan2(ss, cc)))
    
    # 6. 重新分配 t 和更新 psi
    for i in range(m):
        out_pts[i]['psi'] = psi_sm[i]
        out_pts[i]['t'] = i * dt
    
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
    ap.add_argument('--smooth-window', type=int, default=5, help='psi smoothing window size (odd)')
    ap.add_argument('--smooth-factor', type=float, default=None, 
                    help='Spline smoothing factor. 0=Interpolate (bad!). Default: m (number of points)')
    ap.add_argument('--num-factor', type=int, default=20, 
                    help='Density multiplier (e.g., 20 means 20x original point count)')
    ap.add_argument('--no_densify', action='store_true', help='disable densify/resample steps')
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
        
        # 【V4 修正点】现在 convert_group 会正确排序
        pts = convert_group(rows, heading_ref=args.heading_ref, dt=args.dt)
        
        if not pts:
            print(f'No valid, sortable points found in {path}')
            return

        if args.normalize:
            pts = rotate_normalize(pts)
            
        if not args.no_densify:
            smooth_factor = args.smooth_factor
            if smooth_factor is None:
                smooth_factor = len(pts)  # 默认 s = m (点数)
            
            # (V3 函数 - 保留)
            # 现在这个函数接收到的是已排序的 'pts'
            pts = resample_with_smoothing_spline(pts,
                                       dup_eps=args.dup_eps,
                                       smooth_factor=smooth_factor,
                                       num_factor=args.num_factor,
                                       dt=args.dt,
                                       smooth_window=args.smooth_window)

        out_path = os.path.join(outdir_local, f'plan_{stem}.csv')
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