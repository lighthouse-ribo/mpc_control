import json
import numpy as np
import os
from typing import Any, Dict

# 可选 YAML 支持：优先 ruamel.yaml（保留注释），其次 PyYAML；都不可用时回退 JSON
_HAS_RUAMEL = False
_HAS_PYYAML = False
try:
    from ruamel.yaml import YAML  # type: ignore
    _HAS_RUAMEL = True
except Exception:
    try:
        import yaml  # type: ignore
        _HAS_PYYAML = True
    except Exception:
        pass

DEFAULT_CONFIG: Dict[str, Any] = {
    "vehicle": {
        "m": 35000.0,
        "Iz": 500000.0,
        "a": 8.0,
        "b": 8.0,
        "width": 3.4,
        "track": 3.4,
        "kf": 450000.0,
        "kr": 450000.0,
        "U": 2.2,
        "mu": 0.85,
        "g": 9.81,
        "U_min": 0.001,
        "U_blend": 0.3,
        "tire_model": "pacejka",
    },
    "control": {
        "dt": 0.02,
        "k_v": 0.8,
        "tau_ctrl": 0.15,
        "tau_low": 0.25,
        "tau_beta": 0.35,
        "yaw_damp": 220.0,
        "yaw_sat_gain": 3.0,
        "drive_bias_front": 0.1,
        "drive_bias_rear": 0.9,
        "delta_rate_frac": 0.8,
        "delta_max": 0.698131700797,  # 40 deg in rad
        "U_switch": 8.0,
        "phase_auto": False,
        "mode": "2dof",
        "track": {"enabled": True, "retention_sec": 30.0, "max_points": 20000},
    },
}

def load_config(path: str) -> Dict[str, Any]:
    """加载配置：支持 .yaml/.yml/.json；失败时返回 DEFAULT_CONFIG 深拷贝。"""
    if path and os.path.exists(path):
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".yaml", ".yml"):
                if _HAS_RUAMEL:
                    yaml = YAML(typ="rt")  # round-trip 保留注释
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.load(f) or {}
                    # ruamel 返回 CommentedMap，转普通 dict
                    return json.loads(json.dumps(data))
                elif _HAS_PYYAML:
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    return json.loads(json.dumps(data))
                else:
                    # 无 YAML 解析库，回退尝试 JSON 读取
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
    # 深拷贝默认配置
    return json.loads(json.dumps(DEFAULT_CONFIG))

def save_config(path: str, cfg: Dict[str, Any]) -> None:
    """保存配置：按扩展名使用 YAML 或 JSON。注意：普通保存不会保留 YAML 注释。"""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            if _HAS_RUAMEL:
                yaml = YAML()
                yaml.default_flow_style = False
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)
                return
            elif _HAS_PYYAML:
                import yaml as _yaml  # type: ignore
                with open(path, "w", encoding="utf-8") as f:
                    _yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
                return
        # 默认 JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def apply_config(cfg: Dict[str, Any], vp: Any, engine: Any) -> None:
    # vehicle params
    vcfg = cfg.get("vehicle") or {}
    for k in ("m","Iz","a","b","width","track","kf","kr","U","mu","g","U_min","U_blend"):
        if k in vcfg:
            try:
                setattr(vp, k, float(vcfg[k]))
            except (TypeError, ValueError):
                pass
    model = vcfg.get("tire_model")
    if isinstance(model, str) and model.lower() in ("pacejka","linear"):
        vp.tire_model = model.lower()

    # control config applied to engine
    ccfg = cfg.get("control") or {}
    for k in ("k_v","tau_ctrl","tau_low","tau_beta","yaw_damp","yaw_sat_gain","drive_bias_front","drive_bias_rear","delta_rate_frac","delta_max","U_switch"):
        if k in ccfg:
            try:
                setattr(engine, k, float(ccfg[k]))
            except (TypeError, ValueError):
                pass
    if "phase_auto" in ccfg:
        try:
            engine.phase_auto = bool(ccfg["phase_auto"])
        except Exception:
            pass
    # mode
    mode = ccfg.get("mode")
    if isinstance(mode, str) and mode in ("2dof","3dof"):
        try:
            engine.set_mode(mode)
        except Exception:
            pass
    # track settings
    tcfg = ccfg.get("track") or {}
    if isinstance(tcfg, dict):
        engine.set_track_settings(
            enabled=tcfg.get("enabled"),
            retention_sec=tcfg.get("retention_sec"),
            max_points=tcfg.get("max_points"),
        )
    # sync control U
    try:
        engine.set_ctrl(U=vp.U)
    except Exception:
        pass

def current_config(vp: Any, engine: Any) -> Dict[str, Any]:
    # Reflect current values back into config structure
    vehicle = {
        "m": float(vp.m),
        "Iz": float(vp.Iz),
        "a": float(vp.a),
        "b": float(vp.b),
        "width": float(vp.width),
        "track": float(vp.track),
        "kf": float(vp.kf),
        "kr": float(vp.kr),
        "U": float(vp.U),
        "mu": float(vp.mu),
        "g": float(vp.g),
        "U_min": float(vp.U_min),
        "U_blend": float(getattr(vp, "U_blend", 0.3)),
        "tire_model": str(vp.tire_model),
    }
    control = {
        "dt": float(getattr(engine, "dt", 0.02)),
        "k_v": float(getattr(engine, "k_v", 0.8)),
        "tau_ctrl": float(getattr(engine, "tau_ctrl", 0.15)),
        "tau_low": float(getattr(engine, "tau_low", 0.25)),
        "tau_beta": float(getattr(engine, "tau_beta", 0.35)),
        "yaw_damp": float(getattr(engine, "yaw_damp", 220.0)),
        "yaw_sat_gain": float(getattr(engine, "yaw_sat_gain", 3.0)),
        "drive_bias_front": float(getattr(engine, "drive_bias_front", 0.1)),
        "drive_bias_rear": float(getattr(engine, "drive_bias_rear", 0.9)),
        "delta_rate_frac": float(getattr(engine, "delta_rate_frac", 0.8)),
        "delta_max": float(getattr(engine, "delta_max", 0.698131700797)),
        "U_switch": float(getattr(engine, "U_switch", 8.0)),
        "phase_auto": bool(getattr(engine, "phase_auto", False)),
        "mode": str(engine.get_ctrl().get("mode", "2dof")),
        "track": {
            "enabled": bool(engine.track_cfg.enabled),
            "retention_sec": float(engine.track_cfg.retention_sec),
            "max_points": int(engine.track_cfg.max_points),
        },
    }
    return {"vehicle": vehicle, "control": control}

# src/config.py (部分)

class PlanningConfig:
    """路径规划专用配置"""
    def __init__(self):
        # A* 核心参数 (完全对齐 airportPathPlanning.py)
        self.dynamic_turn_cost = True
        self.safety_margin = 2.5      # 原始值 1.7 (工程单位: 米)
        self.turn_cost_weight = 4.0
        self.min_turn_angle = 13.0
        self.max_turn_angle = 50.0
        
        # 平滑参数
        self.use_bspline_smoothing = True
        self.smoothing_factor = 10
