import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class InitParams:
    initial_pose: List[float]
    initial_factor: float
    k_estimator_mode: bool
    default_k: float
    tac3d_name1: str
    tac3d_name2: str
    grasping_force: float

@dataclass
class RandomErrorParams:
    random_seed: int
    x_scale: float
    y_scale: float
    z_scale: float
    roll_scale: float
    pitch_scale: float
    yaw_scale: float

@dataclass
class Config:
    init_params: InitParams
    random_error_params: RandomErrorParams

def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config(
        init_params=InitParams(**config_dict['init_params']),
        random_error_params=RandomErrorParams(**config_dict['random_error'])
    )

# 全局变量，存储加载的配置
_config = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config