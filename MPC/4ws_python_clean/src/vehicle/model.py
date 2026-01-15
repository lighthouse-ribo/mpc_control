from dataclasses import dataclass


@dataclass
class SimState:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0  # rad
    beta: float = 0.0
    r: float = 0.0


@dataclass
class Control:
    U: float = 10.0           # m/s
    delta_f: float = 0.0      # rad
    delta_r: float = 0.0      # rad


@dataclass
class TrackSettings:
    enabled: bool = True
    retention_sec: float = 30.0
    max_points: int = 20000