from flask import Flask, jsonify, request
from flask import send_from_directory
from src.vehicle import VehicleParams
from src.sim import SimEngine
from src.map_server import MapServer
from src.config import PlanningConfig
from src.Global_planning import EnhancedAStar
from src.lattice_planning.lattice_planner import LatticePlanner
from src.config import load_config, apply_config, current_config, save_config
from src.planner import plan_quintic_xy, plan_circle_arc
import math
import numpy as np
import os

app = Flask(__name__, static_folder='web', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# 启动时加载配置文件并应用
def _resolve_cfg_path() -> str:
    env_path = os.environ.get("SIM_CONFIG_PATH")
    if env_path:
        return env_path
    # 优先使用 YAML（若存在），否则回退 JSON
    yaml_path = os.path.join(os.getcwd(), "config.yaml")
    json_path = os.path.join(os.getcwd(), "config.json")
    if os.path.exists(yaml_path):
        return yaml_path
    return json_path

CFG_PATH = _resolve_cfg_path()
_CFG = load_config(CFG_PATH)
VP = VehicleParams()
ENGINE = SimEngine(VP, dt=float((_CFG.get("control") or {}).get("dt", 0.02)))

# [已存在] 初始化规划配置与地图服务
PLAN_CFG = PlanningConfig()
MAP_SERVER = MapServer() 

# [新增] 初始化 A* 规划器
ASTAR_PLANNER = EnhancedAStar(PLAN_CFG)
LATTICE_PLANNER = LatticePlanner()

apply_config(_CFG, VP, ENGINE)

# 静态首页与静态资源
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)

# 获取/更新车辆参数
@app.route('/api/params', methods=['GET', 'POST', 'PATCH'])
def api_params():
    if request.method == 'GET':
        return jsonify(VP.to_dict())
    data = request.get_json(force=True) or {}
    if request.method == 'POST':
        # 兼容旧参数平铺结构
        for k in ['m', 'Iz', 'a', 'b', 'width', 'track', 'kf', 'kr', 'U', 'mu', 'g', 'U_min', 'U_blend']:
            if k in data:
                try:
                    setattr(VP, k, float(data[k]))
                except (TypeError, ValueError):
                    pass
        # 非数值参数：轮胎模型选择
        if 'tire_model' in data:
            v = str(data['tire_model']).lower().strip()
            if v in ('pacejka', 'linear'):
                VP.tire_model = v
        # 若更新了 U，则同步到控制
        ENGINE.set_ctrl(U=VP.U)
        # 持久化到配置文件
        save_config(CFG_PATH, current_config(VP, ENGINE))
        return jsonify(VP.to_dict())
    else:
        # PATCH：支持嵌套结构（tire/control），并兼容平铺更新
        # 平铺部分（与 POST 相同）
        for k in ['m', 'Iz', 'a', 'b', 'width', 'track', 'kf', 'kr', 'U', 'mu', 'g', 'U_min', 'U_blend']:
            if k in data:
                try:
                    setattr(VP, k, float(data[k]))
                except (TypeError, ValueError):
                    pass
        if 'tire_model' in data:
            v = str(data['tire_model']).lower().strip()
            if v in ('pacejka', 'linear'):
                VP.tire_model = v

        # 嵌套：tire
        tire = data.get('tire') or {}
        if isinstance(tire, dict):
            # model
            model = tire.get('model')
            if isinstance(model, str):
                v = model.lower().strip()
                if v in ('pacejka', 'linear'):
                    VP.tire_model = v
            # lateral.mu_y
            lat = tire.get('lateral') or {}
            if isinstance(lat, dict) and 'mu_y' in lat:
                try:
                    VP.mu = float(lat['mu_y'])
                except (TypeError, ValueError):
                    pass
            # longitudinal.mu_x（当前与 mu 统一映射，保留字段以兼容扩展）
            lon = tire.get('longitudinal') or {}
            if isinstance(lon, dict) and 'mu_x' in lon:
                try:
                    # 简化：先与 mu 同步（未来可拆分为独立字段）
                    VP.mu = float(lon['mu_x'])
                except (TypeError, ValueError):
                    pass

        # 嵌套：control（仿真控制/配置）
        control = data.get('control') or {}
        if isinstance(control, dict):
            for k in ['k_v', 'tau_ctrl', 'tau_low', 'tau_beta', 'yaw_damp', 'yaw_sat_gain', 'drive_bias_front', 'drive_bias_rear', 'delta_rate_frac']:
                if k in control:
                    try:
                        val = float(control[k])
                        if k == 'k_v':
                            ENGINE.k_v = val
                        elif k == 'tau_ctrl':
                            ENGINE.tau_ctrl = val
                        elif k == 'tau_low':
                            ENGINE.tau_low = val
                        elif k == 'tau_beta':
                            ENGINE.tau_beta = val
                        elif k == 'yaw_damp':
                            ENGINE.yaw_damp = val
                        elif k == 'yaw_sat_gain':
                            ENGINE.yaw_sat_gain = val
                        elif k == 'drive_bias_front':
                            ENGINE.drive_bias_front = val
                        elif k == 'drive_bias_rear':
                            ENGINE.drive_bias_rear = val
                        elif k == 'delta_rate_frac':
                            ENGINE.delta_rate_frac = val
                    except (TypeError, ValueError):
                        pass

        # 若更新了 U，则同步到控制
        ENGINE.set_ctrl(U=VP.U)
        # 持久化到配置文件
        save_config(CFG_PATH, current_config(VP, ENGINE))
        return jsonify(VP.to_dict())

# 派生量与横摆率指令（基于当前参数与输入前轮转角）
@app.route('/api/derived', methods=['GET'])
def api_derived():
    try:
        df = float(request.args.get('delta_f', 0.0))
    except (TypeError, ValueError):
        df = 0.0
    return jsonify({
        'K': VP.understeer_gradient(),
        'r_ref': VP.r_ref(df),
        'r_cmd': VP.yaw_rate_cmd(df),
    })

# 状态查询（x, y, psi, beta, r）
@app.route('/api/state', methods=['GET'])
def api_state():
    return jsonify(ENGINE.get_state())

# 遥测采样：提供前端无法直接获取的数据（含导数）
@app.route('/api/telemetry', methods=['GET'])
def api_telemetry():
    st = ENGINE.get_state()
    ct = ENGINE.get_ctrl()
    # 统一返回所需字段：u、坐标、航向角、前后轮角、beta_dot、r_dot
    return jsonify({
        'U': float(ct.get('U', 0.0)),
        'x': float(st.get('x', 0.0)),
        'y': float(st.get('y', 0.0)),
        'psi': float(st.get('psi', 0.0)),
        'df': float(st.get('df', 0.0)),
        'dr': float(st.get('dr', 0.0)),
        'beta_dot': float(st.get('beta_dot', 0.0)),
        'r_dot': float(st.get('r_dot', 0.0)),
        # 备用：提供实际速度与半径，便于分析（列不强制）
        'speed': float(st.get('speed', ct.get('U', 0.0))),
        'radius': st.get('radius') if st.get('radius') is not None else None,
        # 性能：自动控制段耗时、循环耗时与睡眠、实际频率与目标 dt
        'autop_ms': float(getattr(ENGINE, '_last_autop_ms', 0.0)),
        'loop_ms': float(getattr(ENGINE, '_last_loop_ms', 0.0)),
        'sleep_ms': float(getattr(ENGINE, '_last_sleep_ms', 0.0)),
        'tick_hz': float(getattr(ENGINE, '_last_tick_hz', 0.0)),
        'dt_cfg': float(getattr(ENGINE, 'dt', 0.0)),
    })

# 控制量查询/更新（U, df, dr, running）
@app.route('/api/control', methods=['GET', 'POST'])
def api_control():
    if request.method == 'GET':
        return jsonify(ENGINE.get_ctrl())
    data = request.get_json(force=True) or {}
    # 角度限幅（更符合直觉的默认范围，单位：度）
    ANGLE_LIMIT_DEG = 30.0
    ANGLE_LIMIT_RAD = np.deg2rad(ANGLE_LIMIT_DEG)

    # 解析角度输入：优先 df_deg/dr_deg；其次 df_rad/dr_rad；最后 df/dr（按“度”理解）
    def parse_angle_rad(d: dict, base: str):
        val = d.get(f"{base}_deg")
        if val is not None:
            try:
                return float(val) * np.pi / 180.0
            except (TypeError, ValueError):
                return None
        val = d.get(f"{base}_rad")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        val = d.get(base)
        if val is not None:
            try:
                # 为直觉一致，df/dr 默认为度
                return float(val) * np.pi / 180.0
            except (TypeError, ValueError):
                return None
        return None

    # 更新 U、df、dr，并做角度限幅
    U = data.get('U')
    df_rad = parse_angle_rad(data, 'df')
    dr_rad = parse_angle_rad(data, 'dr')
    if df_rad is not None:
        df_rad = float(np.clip(df_rad, -ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD))
    if dr_rad is not None:
        dr_rad = float(np.clip(dr_rad, -ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD))

    ENGINE.set_ctrl(U=U if U is not None else None,
                    df=df_rad,
                    dr=dr_rad)
    # 同步到参数的 U（用于派生量计算）
    if U is not None:
        try:
            VP.U = float(U)
        except (TypeError, ValueError):
            pass
    # 持久化到配置文件
    save_config(CFG_PATH, current_config(VP, ENGINE))
    return jsonify(ENGINE.get_ctrl())

# 仿真开始/暂停
@app.route('/api/sim/start_pause', methods=['POST'])
def api_start_pause():
    data = request.get_json(force=True) or {}
    running = data.get('running')
    if isinstance(running, bool):
        if running:
            ENGINE.start()
        else:
            ENGINE.pause()
    else:
        running = ENGINE.toggle()
    return jsonify({'running': bool(running)})

# 仿真重置
@app.route('/api/sim/reset', methods=['POST'])
def api_reset():
    # 重置仿真：同时禁用自动跟踪、清空参考计划，并将前后轮角归零
    ENGINE.reset()
    try:
        ENGINE.set_autop(False)
    except Exception:
        pass
    try:
        # 清空参考轨迹（防止重置后立即继续自动跟踪）
        ENGINE.load_plan([])
    except Exception:
        pass
    try:
        ENGINE.set_ctrl(df=0.0, dr=0.0)
    except Exception:
        pass
    return jsonify({'ok': True})

# 设置初始位姿
@app.route('/api/init_pose', methods=['POST'])
def api_init_pose():
    data = request.get_json(force=True) or {}
    def parse_float(name, default=0.0):
        try:
            return float(data.get(name, default))
        except (TypeError, ValueError):
            return default
    x0 = parse_float('x', 0.0)
    y0 = parse_float('y', 0.0)
    psi = data.get('psi')
    # 接收度或弧度
    psi_rad = None
    try:
        v = float(psi)
        psi_rad = v * np.pi / 180.0 if abs(v) > np.pi else v
    except (TypeError, ValueError):
        psi_rad = 0.0
    ENGINE.set_init_pose(x0, y0, psi_rad)
    return jsonify({'ok': True})

# 轨迹查询与设置
@app.route('/api/track', methods=['GET'])
def api_track():
    return jsonify({'points': ENGINE.get_track()})

@app.route('/api/track/settings', methods=['POST'])
def api_track_settings():
    data = request.get_json(force=True) or {}
    enabled = data.get('enabled')
    keep = data.get('retentionSec')
    maxp = data.get('maxPoints')
    ENGINE.set_track_settings(enabled=enabled, retention_sec=keep, max_points=maxp)
    # 持久化到配置文件
    save_config(CFG_PATH, current_config(VP, ENGINE))
    return jsonify({'ok': True})

# 参考规划轨迹查询（points: [{t,x,y,psi}]）
@app.route('/api/plan', methods=['GET'])
def api_plan_get():
    try:
        pts = getattr(ENGINE, 'plan', []) or []
        try:
            maxn = int(request.args.get('max', '0'))
        except Exception:
            maxn = 0
        if maxn and maxn > 0 and len(pts) > maxn:
            n = len(pts)
            out = []
            for i in range(maxn):
                idx = int(round(i * (n - 1) / max(1, maxn - 1)))
                out.append(pts[idx])
            pts = out
        return jsonify({'points': pts})
    except Exception as e:
        return jsonify({'error': str(e), 'points': []}), 200

@app.route('/api/plan/meta', methods=['GET'])
def api_plan_meta():
    try:
        pts = getattr(ENGINE, 'plan', []) or []
        return jsonify({'count': len(pts)})
    except Exception as e:
        return jsonify({'error': str(e)}), 200

@app.route('/api/plan/chunk', methods=['GET'])
def api_plan_chunk():
    try:
        pts = getattr(ENGINE, 'plan', []) or []
        try:
            start = int(request.args.get('start', '0'))
            count = int(request.args.get('count', '1000'))
        except Exception:
            start = 0; count = 1000
        start = max(0, start)
        end = min(len(pts), start + max(0, count))
        return jsonify({'points': pts[start:end], 'start': start, 'end': end, 'total': len(pts)})
    except Exception as e:
        return jsonify({'error': str(e), 'points': []}), 200

# 五次多项式规划：起点-终点（含航向角）生成参考轨迹
@app.route('/api/plan/quintic', methods=['POST'])
def api_plan_quintic():
    data = request.get_json(force=True) or {}

    def parse_pose(obj, fallback=None):
        if not isinstance(obj, dict):
            return fallback or {'x': 0.0, 'y': 0.0, 'psi': 0.0}
        def parse_val(k, d=0.0):
            try:
                return float(obj.get(k, d))
            except (TypeError, ValueError):
                return d
        x = parse_val('x', (fallback or {}).get('x', 0.0))
        y = parse_val('y', (fallback or {}).get('y', 0.0))
        psi_raw = obj.get('psi', (fallback or {}).get('psi', 0.0))
        psi_rad = 0.0
        try:
            v = float(psi_raw)
            psi_rad = v * np.pi / 180.0 if abs(v) > np.pi else v
        except (TypeError, ValueError):
            psi_rad = float((fallback or {}).get('psi', 0.0))
        return {'x': x, 'y': y, 'psi': psi_rad}

    # 起点：优先使用传入的 start，否则用当前仿真状态
    st = ENGINE.get_state()
    start_default = {'x': float(st['x']), 'y': float(st['y']), 'psi': float(st['psi'])}
    start = parse_pose(data.get('start'), start_default)

    # 终点：必须传入
    end = parse_pose(data.get('end'), None)
    if end is None:
        return jsonify({'error': 'missing end pose'}), 400

    # 规划时长与采样数
    # 规划时长 T 已移除：后端自动估算（不再从请求读取）
    dist = float(np.hypot(end['x'] - start['x'], end['y'] - start['y']))
    U = float(VP.U)
    U = max(0.3, abs(U))
    T = max(1.0, dist / U)
    T = min(T, 30.0)
    try:
        N = int(data.get('N', 200))
    except (TypeError, ValueError):
        N = 200
    N = max(20, N)

    # 规划
    plan = plan_quintic_xy(start, end, T, N, U_start=float(VP.U))
    ENGINE.load_plan(plan)
    return jsonify({'ok': True, 'N': N, 'count': len(plan)})

# 圆弧规划：以起点/终点位置与航向角生成圆弧参考轨迹
@app.route('/api/plan/circle', methods=['POST'])
def api_plan_circle():
    data = request.get_json(force=True) or {}

    def parse_pose(obj, fallback=None):
        if not isinstance(obj, dict):
            return fallback or {'x': 0.0, 'y': 0.0, 'psi': 0.0}
        def parse_val(k, d=0.0):
            try:
                return float(obj.get(k, d))
            except (TypeError, ValueError):
                return d
        x = parse_val('x', (fallback or {}).get('x', 0.0))
        y = parse_val('y', (fallback or {}).get('y', 0.0))
        psi_raw = obj.get('psi', (fallback or {}).get('psi', 0.0))
        psi_rad = 0.0
        try:
            v = float(psi_raw)
            psi_rad = v * np.pi / 180.0 if abs(v) > np.pi else v
        except (TypeError, ValueError):
            psi_rad = float((fallback or {}).get('psi', 0.0))
        return {'x': x, 'y': y, 'psi': psi_rad}

    # 起点：优先使用传入的 start，否则用当前仿真状态
    st = ENGINE.get_state()
    start_default = {'x': float(st['x']), 'y': float(st['y']), 'psi': float(st['psi'])}
    start = parse_pose(data.get('start'), start_default)

    # 终点：必须传入
    end = parse_pose(data.get('end'), None)
    if end is None:
        return jsonify({'error': 'missing end pose'}), 400

    # 采样数（圆弧内部自用 T，外部传入 T 作为时间标签均匀分布）
    dist = float(np.hypot(end['x'] - start['x'], end['y'] - start['y']))
    U = float(VP.U)
    U = max(0.3, abs(U))
    T = max(1.0, dist / U)
    T = min(T, 30.0)
    try:
        N = int(data.get('N', 200))
    except (TypeError, ValueError):
        N = 200
    N = max(20, N)

    plan = plan_circle_arc(start, end, T, N, U_start=float(VP.U))
    ENGINE.load_plan(plan)
    return jsonify({'ok': True, 'N': N, 'count': len(plan)})

@app.route('/api/plan/import', methods=['POST', 'OPTIONS'])
def api_plan_import():
    if request.method == 'OPTIONS':
        return ('', 204)
    # 兼容 multipart 上传：若存在文件则走 CSV 路径
    try:
        if 'file' in request.files:
            f = request.files['file']
            if not f:
                return jsonify({'error': 'empty upload'}), 400
            from io import TextIOWrapper
            import csv as _csv
            wrapper = TextIOWrapper(f.stream, encoding='utf-8', errors='replace')
            reader = _csv.reader(wrapper)
            header = next(reader, None)
            if not header or [h.strip() for h in header] != ['t','x','y','psi']:
                return jsonify({'error': 'header must be t,x,y,psi'}), 400
            pts = []
            for row in reader:
                if not row or len(row) < 4:
                    continue
                try:
                    t = float(row[0]) if row[0] != '' else 0.0
                    x = float(row[1]); y = float(row[2])
                    psi = float(row[3]) if row[3] != '' else 0.0
                except Exception:
                    return jsonify({'error': 'invalid numeric row'}), 400
                pts.append({'t': t, 'x': x, 'y': y, 'psi': psi})
            if len(pts) < 2:
                return jsonify({'error': 'too few points'}), 400
            # 简易清洗与重算 psi
            out = []
            last = None
            for p in pts:
                if last is None:
                    out.append(p); last = p; continue
                dx = p['x'] - last['x']; dy = p['y'] - last['y']
                ds = float(np.hypot(dx, dy))
                if ds <= 1e-6:
                    continue
                out.append(p); last = p
            for i in range(len(out)):
                if i < len(out) - 1:
                    dx = out[i+1]['x'] - out[i]['x']; dy = out[i+1]['y'] - out[i]['y']
                else:
                    dx = out[i]['x'] - out[i-1]['x']; dy = out[i]['y'] - out[i-1]['y']
                out[i]['psi'] = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-9 else out[i].get('psi', 0.0)
            W = 5; half = W // 2
            psi_sm = []
            for i in range(len(out)):
                a = max(0, i - half); b = min(len(out) - 1, i + half)
                ss = 0.0; cc = 0.0
                for k in range(a, b + 1):
                    ss += float(np.sin(out[k]['psi']))
                    cc += float(np.cos(out[k]['psi']))
                psi_sm.append(float(np.arctan2(ss, cc)))
            for i in range(len(out)):
                out[i]['psi'] = psi_sm[i]
                out[i]['t'] = float(i) * 0.05
            ENGINE.load_plan(out)
            return jsonify({'ok': True, 'count': len(out)})
    except Exception:
        pass
    # JSON points 路径
    data = request.get_json(force=True) or {}
    pts = data.get('points')
    if not isinstance(pts, list):
        return jsonify({'error': 'points must be array'}), 400
    n = len(pts)
    if n < 2:
        return jsonify({'error': 'points length must be >= 2'}), 400
    opts = data.get('options') if isinstance(data.get('options'), dict) else {}
    try:
        dup_eps = float(opts.get('dup_eps', 1e-3))
    except Exception:
        dup_eps = 1e-3
    try:
        max_step = float(opts.get('max_step', 1.0))
    except Exception:
        max_step = 1.0
    try:
        smooth_window = int(opts.get('smooth_window', 5))
    except Exception:
        smooth_window = 5
    try:
        dt_val = float(opts.get('dt', 0.1))
    except Exception:
        dt_val = 0.1

    def _wrap(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    out = []
    for i in range(n):
        p = pts[i] if isinstance(pts[i], dict) else {}
        try:
            x = float(p.get('x'))
            y = float(p.get('y'))
        except (TypeError, ValueError):
            return jsonify({'error': f'invalid x/y at index {i}'}), 400
        t_raw = p.get('t', None)
        psi_raw = p.get('psi', None)
        t_val = None
        psi_val = None
        try:
            if t_raw is not None:
                t_val = float(t_raw)
        except (TypeError, ValueError):
            return jsonify({'error': f'invalid t at index {i}'}), 400
        try:
            if psi_raw is not None:
                psi_val = _wrap(float(psi_raw))
        except (TypeError, ValueError):
            return jsonify({'error': f'invalid psi at index {i}'}), 400
        out.append({'t': (t_val if t_val is not None else 0.0), 'x': x, 'y': y, 'psi': (psi_val if psi_val is not None else 0.0)})

    for i in range(n):
        if not np.isfinite(out[i]['psi']):
            out[i]['psi'] = 0.0
    cleaned = []
    removed_dup = 0
    for i in range(len(out)):
        if i == 0:
            cleaned.append(out[i])
            continue
        dx = out[i]['x'] - cleaned[-1]['x']
        dy = out[i]['y'] - cleaned[-1]['y']
        ds = float(np.hypot(dx, dy))
        if ds <= dup_eps:
            removed_dup += 1
            continue
        cleaned.append(out[i])
    if len(cleaned) < 2:
        return jsonify({'error': 'trajectory too short after duplicate removal'}), 400
    inserted = 0
    densified = []
    for i in range(len(cleaned) - 1):
        p0 = cleaned[i]
        p1 = cleaned[i + 1]
        densified.append(p0)
        dx = p1['x'] - p0['x']
        dy = p1['y'] - p0['y']
        ds = float(np.hypot(dx, dy))
        if ds > max_step:
            k = int(np.ceil(ds / max_step))
            if k > 1:
                for j in range(1, k):
                    r = j / k
                    xi = p0['x'] + r * dx
                    yi = p0['y'] + r * dy
                    densified.append({'t': 0.0, 'x': xi, 'y': yi, 'psi': 0.0})
                    inserted += 1
    densified.append(cleaned[-1])
    m = len(densified)
    if m < 2:
        return jsonify({'error': 'trajectory too short after densify'}), 400
    for i in range(m):
        if i < m - 1:
            dx = densified[i + 1]['x'] - densified[i]['x']
            dy = densified[i + 1]['y'] - densified[i]['y']
        else:
            dx = densified[i]['x'] - densified[i - 1]['x']
            dy = densified[i]['y'] - densified[i - 1]['y']
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            densified[i]['psi'] = densified[i - 1]['psi'] if i > 0 else 0.0
        else:
            densified[i]['psi'] = _wrap(float(np.arctan2(dy, dx)))
    if smooth_window < 3:
        smooth_window = 3
    if smooth_window % 2 == 0:
        smooth_window += 1
    half = smooth_window // 2
    psi_sm = []
    for i in range(m):
        a = max(0, i - half)
        b = min(m - 1, i + half)
        ss = 0.0
        cc = 0.0
        for k in range(a, b + 1):
            ss += float(np.sin(densified[k]['psi']))
            cc += float(np.cos(densified[k]['psi']))
        psi_sm.append(_wrap(float(np.arctan2(ss, cc))))
    for i in range(m):
        densified[i]['psi'] = psi_sm[i]
        densified[i]['t'] = float(i) * dt_val

    ENGINE.load_plan(densified)
    return jsonify({'ok': True, 'count': len(densified), 'inserted_points': inserted, 'removed_duplicates': removed_dup})

# 自动跟踪开关/查询（根据参考轨迹沿路输出前后轮角）
@app.route('/api/autop', methods=['GET', 'POST'])
def api_autop():
    if request.method == 'GET':
        # 返回自动状态与设备信息
        
        # 1. 获取控制器类型：优先从引擎获取，默认回退到 'mpc' (原为 'simple')
        controller_type = 'mpc'
        if hasattr(ENGINE, 'get_controller_type'):
            controller_type = ENGINE.get_controller_type()
        else:
            controller_type = getattr(ENGINE, 'autop_mode', 'mpc')

        # 2. 安全获取 plan_count
        plan = getattr(ENGINE, 'plan', [])
        plan_count = len(plan) if plan is not None else 0

        return jsonify({
            'enabled': bool(getattr(ENGINE, 'autop_enabled', False)),
            'mode': getattr(ENGINE, 'autop_mode', 'mpc'),  # <--- 修改：默认改为 mpc
            'controller': controller_type,
            'plan_count': plan_count,
            'device': 'cpu',     
            'device_active': False,
        })
    
    # POST 请求处理
    data = request.get_json(force=True) or {}
    
    # 设置启用状态
    enabled = bool(data.get('enabled', True))
    if hasattr(ENGINE, 'set_autop'):
        ENGINE.set_autop(enabled)
    
    # 设置模式
    mode = data.get('mode')
    if isinstance(mode, str):
        try:
            if hasattr(ENGINE, 'set_autop_mode'):
                ENGINE.set_autop_mode(mode)
        except Exception:
            pass
            
    # 可选：启动仿真
    if bool(data.get('start', False)):
        if hasattr(ENGINE, 'start'):
            ENGINE.start()
        
    # 构建返回数据 (逻辑同 GET)
    controller_type = 'mpc'
    if hasattr(ENGINE, 'get_controller_type'):
        controller_type = ENGINE.get_controller_type()
    else:
        controller_type = getattr(ENGINE, 'autop_mode', 'mpc') # <--- 修改：默认改为 mpc

    return jsonify({
        'enabled': bool(getattr(ENGINE, 'autop_enabled', False)),
        'mode': getattr(ENGINE, 'autop_mode', 'mpc'), # <--- 修改：默认改为 mpc
        'controller': controller_type,
        'device': 'cpu',      
        'device_active': False,
    })

# 仿真模式：2DOF / 3DOF
@app.route('/api/mode', methods=['GET', 'POST'])
def api_mode():
    if request.method == 'GET':
        return jsonify({'mode': ENGINE.get_ctrl().get('mode', '2dof')})
    data = request.get_json(force=True) or {}
    mode = data.get('mode')
    ENGINE.set_mode(mode if isinstance(mode, str) else '2dof')
    # 持久化到配置文件
    save_config(CFG_PATH, current_config(VP, ENGINE))
    return jsonify({'mode': ENGINE.get_ctrl().get('mode', '2dof')})

# 配置查询（完整 vehicle/control 配置）
@app.route('/api/config', methods=['GET'])
def api_config_get():
    try:
        return jsonify(current_config(VP, ENGINE))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plan/import_csv', methods=['POST', 'OPTIONS'])
def api_plan_import_csv():
    # --- [修改] 将导入语句移到函数内部，防止 NameError ---
    from io import TextIOWrapper
    import csv as _csv
    # ------------------------------------------------
    
    try:
        if request.method == 'OPTIONS':
            return ('', 204)
        if 'file' not in request.files:
            return jsonify({'error': 'missing file field'}), 400
        f = request.files['file']
        if not f:
            return jsonify({'error': 'empty upload'}), 400
        # 流式读取 CSV
        wrapper = TextIOWrapper(f.stream, encoding='utf-8', errors='replace')
        reader = _csv.reader(wrapper)
        header = next(reader, None)
        if not header or [h.strip() for h in header] != ['t','x','y','psi']:
            return jsonify({'error': 'header must be t,x,y,psi'}), 400
        pts = []
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                t = float(row[0]) if row[0] != '' else 0.0
                x = float(row[1])
                y = float(row[2])
                psi = float(row[3]) if row[3] != '' else 0.0
            except Exception:
                return jsonify({'error': 'invalid numeric row'}), 400
            pts.append({'t': t, 'x': x, 'y': y, 'psi': psi})
        if len(pts) < 2:
            return jsonify({'error': 'too few points'}), 400
        # 基础清洗与平滑：复用导入逻辑（简化版）
        out = []
        last = None
        for p in pts:
            if last is None:
                out.append(p); last = p; continue
            dx = p['x'] - last['x']; dy = p['y'] - last['y']
            ds = float(np.hypot(dx, dy))
            if ds <= 1e-6:
                continue
            out.append(p); last = p
        # 重算 psi（几何）
        for i in range(len(out)):
            if i < len(out) - 1:
                dx = out[i+1]['x'] - out[i]['x']; dy = out[i+1]['y'] - out[i]['y']
            else:
                dx = out[i]['x'] - out[i-1]['x']; dy = out[i]['y'] - out[i-1]['y']
            out[i]['psi'] = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-9 else out[i].get('psi', 0.0)
        # 圆均值平滑 psi
        W = 5
        half = W // 2
        psi_sm = []
        for i in range(len(out)):
            a = max(0, i - half); b = min(len(out) - 1, i + half)
            ss = 0.0; cc = 0.0
            for k in range(a, b + 1):
                ss += float(np.sin(out[k]['psi']))
                cc += float(np.cos(out[k]['psi']))
            psi_sm.append(float(np.arctan2(ss, cc)))
        for i in range(len(out)):
            out[i]['psi'] = psi_sm[i]
            out[i]['t'] = float(i) * 0.05
        ENGINE.load_plan(out)
        return jsonify({'ok': True, 'count': len(out)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/api/plan/astar', methods=['POST'])
def api_plan_astar():
    """
    [升级版] 规划接口: A* 全局规划 + Lattice 局部轨迹生成
    """
    data = request.get_json(force=True) or {}
    
    # --- 1. 解析起终点 ---
    start_req = data.get('start')
    
    # 2. 获取当前车辆真实状态 (用于 Lattice 初始状态)
    sim_state = ENGINE.get_state()
    
    if not start_req:
        sx, sy = sim_state['x'], sim_state['y']
        # 如果没有指定起点，直接使用车辆当前状态作为 Lattice 起点
        start_state_for_lattice = {
            'x': sim_state['x'],
            'y': sim_state['y'],
            'yaw': sim_state['yaw'],
            'v': sim_state['v'],
            'a': 0.0, # 简化的假设，也可以从 engine 获取
            'k': 0.0  # 简化的假设
        }
    else:
        # 如果用户在大地图上手点了起点 (重置规划)
        sx, sy = float(start_req.get('x')), float(start_req.get('y'))
        # 对于手点的位置，我们假设车辆是静止的，且朝向需要推断
        # (稍后根据 A* 路径推断 yaw)
        start_state_for_lattice = {
            'x': sx, 'y': sy, 'yaw': 0.0, 'v': 0.0, 'a': 0.0, 'k': 0.0
        }
    
    end_req = data.get('end')
    if not end_req:
        return jsonify({'error': 'missing end pose', 'ok': False}), 400
    ex, ey = float(end_req.get('x')), float(end_req.get('y'))

    # --- 2. 执行 A* 全局规划 ---
    sr, sc = MAP_SERVER.world_to_grid(sx, sy)
    er, ec = MAP_SERVER.world_to_grid(ex, ey)
    
    if not (MAP_SERVER.is_valid(sr, sc) and MAP_SERVER.is_valid(er, ec)):
         return jsonify({'error': 'out of map bounds', 'ok': False}), 400

    # 获取 A* 平滑路径 (Reference Line)
    # 注意：这里我们拿到的 ref_path 是 [{'x':, 'y':}, ...]
    ref_path = ASTAR_PLANNER.plan(
        MAP_SERVER.grid, 
        (sr, sc), 
        (er, ec), 
        MAP_SERVER.obs_dist, 
        map_info=MAP_SERVER
    )
    
    if not ref_path or len(ref_path) < 3:
        return jsonify({'error': 'global path planning failed', 'ok': False}), 500

    # --- 3. 准备 Lattice 输入状态 ---
    # 如果是手点起点，我们需要根据 A* 的前几个点推断车辆朝向 yaw
    if start_req:
        p0 = ref_path[0]
        p1 = ref_path[1]
        start_state_for_lattice['yaw'] = math.atan2(p1['y'] - p0['y'], p1['x'] - p0['x'])

    # --- 4. 执行 Lattice 局部规划 ---
    print(f"[Plan] Running Lattice... Start: {start_state_for_lattice}")
    
    final_trajectory = LATTICE_PLANNER.plan(
        grid_map=MAP_SERVER.grid,
        obs_dist=MAP_SERVER.obs_dist,
        start_state=start_state_for_lattice,
        ref_path_points=ref_path,
        resolution=MAP_SERVER.resolution
    )
    
    if not final_trajectory:
        print("[Plan] Lattice failed to generate valid trajectory!")
        # 降级策略：如果 Lattice 失败（比如起点离障碍物太近），
        # 我们可以暂时返回 A* 路径供可视化，但标注 invalid
        return jsonify({'error': 'lattice planning failed (collision or dynamic constraint)', 'ok': False}), 500

    # --- 5. 加载到仿真引擎 ---
    # final_trajectory 包含 t, x, y, v, a, k, psi，正是 MPC 需要的完美格式
    ENGINE.load_plan(final_trajectory)
    
    return jsonify({
        'ok': True,
        'count': len(final_trajectory),
        'path': final_trajectory # 前端会绘制这条带速度信息的轨迹
    })


if __name__ == '__main__':
    try:
        # 允许通过环境变量 PORT 指定端口，默认改为 18000，避免 8000 被占用
        port = int(os.environ.get('PORT', '18000'))
        app.run(host='0.0.0.0', port=port, debug=True)
    finally:
        # 调试退出时关闭引擎线程
        try:
            ENGINE.shutdown()
        except Exception:
            pass
from io import TextIOWrapper
import csv as _csv