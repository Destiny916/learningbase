# scripts/inference_config.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Tuple, Optional
from act_async_infer_distributed_demo.scripts.utils_distributed import log_warning



@dataclass
class InferenceConfig:
    """推理配置基类。

    提供 JSON 加载、运行时更新。
    子类只需定义字段和 FIELD_NAMES / RUNTIME_NAMES。
    """

    FIELD_NAMES: ClassVar[Tuple[str, ...]] = ()
    """所有字段名（白名单过滤用）。"""

    RUNTIME_NAMES: ClassVar[Tuple[str, ...]] = ()
    """Manager 运行时允许修改的字段名（子集）。为空表示允许全部 FIELD_NAMES。"""

    @classmethod
    def from_json_file(cls, json_path: str = "") -> "InferenceConfig":
        """从 JSON 文件加载配置（未出现的字段使用默认值）。"""
        cfg = cls()
        if json_path and os.path.isfile(json_path):
            with open(json_path) as f:
                raw = json.load(f)
            cls._merge_dict(cfg, raw)
        elif json_path:
            log_warning(f"{cls.__name__}: 配置文件不存在，使用默认值: {json_path}")
        return cfg


    def apply_update(self, updates: Dict,
                     field_filter: Optional[Tuple[str, ...]] = None) -> None:
        """白名单过滤更新。

        Args:
            updates: 要更新的 key-value 字典。
            field_filter: 允许更新的字段名元组。默认用 RUNTIME_NAMES（若为空则用 FIELD_NAMES）。
        """
        allowed = set(field_filter or self.RUNTIME_NAMES or self.FIELD_NAMES)
        valid = {k: v for k, v in updates.items() if k in allowed}
        unknown = set(updates) - allowed
        if unknown:
            log_warning(f"{type(self).__name__}: 忽略不可更新字段: {unknown}")
        for k, v in valid.items():
            setattr(self, k, v)


    @classmethod
    def _merge_dict(cls, cfg: "InferenceConfig", raw: dict) -> None:
        """用 dict 值覆盖 dataclass 字段（白名单过滤）。"""
        unknown = set(raw) - set(cls.FIELD_NAMES)
        if unknown:
            log_warning(f"{cls.__name__}: JSON 中包含未知字段（已忽略）: {unknown}")
        for k, v in raw.items():
            if k in cls.FIELD_NAMES:
                setattr(cfg, k, v)



@dataclass
class ClientConfig(InferenceConfig):
    """客户端配置。"""

    # 系统参数：网络/管理
    server_host: str = "192.168.20.99"
    server_port: int = 8889
    manager_port: int = 8889

    # 系统参数：运行模式
    service: bool = False
    """是否以 service 模式运行（等待 manager 下发配置）。"""

    # 系统参数：ROS2 话题
    joint_topic: str = "/feedback/robot_server_state"
    joint_control_topic: str = "/control/joint_position"
    cam_head_left_topic: str = "/camera/left_eye_resize"
    cam_head_right_topic: str = "/camera/right_eye_resize"
    cam_hand_left_topic: str = "/camera_l/color/image_rect_raw"
    cam_hand_right_topic: str = "/camera_r/color/image_rect_raw"
    left_hand_qpos6_topic: str = "/feedback/hand/left"
    right_hand_qpos6_topic: str = "/feedback/hand/right"
    left_gripper_qpos_topic: str = "/feedback/gripper/left"
    right_gripper_qpos_topic: str = "/feedback/gripper/right"
    set_left_hand_qpos6_topic: str = "/control/ee/left"
    set_right_hand_qpos6_topic: str = "/control/ee/right"
    set_left_gripper_qpos_topic: str = "/control/gripper/left"
    set_right_gripper_qpos_topic: str = "/control/gripper/right"

    # 系统参数：输出/调试
    output_dir: str = "./client_output"
    save_exec_action: bool = False
    save_actionchunks: bool = False

    # 推理参数：观测/采集
    control_frequency: float = 10.0
    collect_frequency: float = 10.0
    max_steps: int = 500
    sample_factor: float = 2.0

    # 推理参数：图像
    use_hand_camera: bool = True
    head_target_size: Tuple[int, int] = (640, 360)
    hand_target_size: Tuple[int, int] = (640, 480)
    hand_left_target_size: Tuple[int, int] = (640, 360)
    hand_right_target_size: Tuple[int, int] = (640, 480)

    # 推理参数：动作/模型
    action_horizon: int = 32
    execution_mode: str = "single"
    time_infer: float = 0.0
    chunk_size_threshold: float = 0.0
    use_td: bool = False
    mode: int = 1
    prompt: str = ""

    # 推理参数：末端执行器
    end_effector_type: str = "hand"
    end_effector_position_limit: list = field(default_factory=lambda: [0, 100])

    # 推理参数：原点
    home_position: str = ""


ClientConfig.FIELD_NAMES = tuple(ClientConfig.__dataclass_fields__.keys())
ClientConfig.RUNTIME_NAMES = (
    "server_host", "server_port",
    "control_frequency", "collect_frequency", "max_steps",
    "head_target_size", "hand_target_size", "hand_left_target_size", "hand_right_target_size", "action_horizon",
    "use_hand_camera", "end_effector_type", "end_effector_position_limit",
    "time_infer", "chunk_size_threshold", "sample_factor",
    "prompt", "mode", "use_td",
    "save_exec_action", "save_actionchunks", "home_position"
)


@dataclass
class ServerConfig(InferenceConfig):
    """服务端推理配置"""

    # 系统参数
    pretrained: str = ""
    """预训练模型路径"""
    model_name: str = ""
    """模型名称"""
    precompute_lang_embeddings: Optional[str] = None
    data_type: str = "real"
    service: bool = True
    force_model_reinit: bool = False
    prev_pickle: str = ""

    # 推理参数：动作
    action_horizon: int = 32

    # 推理参数：末端执行器限位
    model_end_effector_limit: list = field(default_factory=lambda: [0, 100])
    hand_sim_max_limit: list = field(
        default_factory=lambda: [87.27, 157.08, 130.9, 130.9, 130.9, 130.9]
    )
    gripper_sim_max_limit: list = field(default_factory=lambda: [100])
    inverse_gripper: bool = False
    is_gripper_bool: bool = False

    # 推理参数：调试 / 数据保存
    save_input: bool = False
    save_vis_images: bool = False
    save_joint_change: bool = False
    use_smooth: bool = True

    # 系统参数：超时
    client_timeout: float = 20.0



ServerConfig.FIELD_NAMES = tuple(ServerConfig.__dataclass_fields__.keys())
# Server 端不限制运行时更新字段（允许所有 FIELD_NAMES），所以 RUNTIME_NAMES 留空



class RequestType(str, Enum):
    """Client 和 Server 之间的请求类型。"""
    OBSERVATION  = "observation"
    GET_ACTIONS  = "get_actions"
    SETUP_CONFIG = "SETUP_CONFIG"
    STOP         = "STOP"
    STATUS       = "STATUS"
    SHUTDOWN     = "SHUTDOWN"


class ResponseKey(str, Enum):
    """响应/请求中的公共 key。"""
    SUCCESS = "success"
    MESSAGE = "message"
    STATUS  = "status"
    STATE   = "state"
    ERROR   = "error"
    CONFIG  = "config"
    TYPE    = "type"


class ManagerKey(str, Enum):
    """Manager 控制协议中的 key。"""
    COMMAND       = "command"
    PAYLOAD       = "payload"
    CLIENT_CONFIG = "client_config"
    SERVER_CONFIG = "server_config"
    HOME_POSITION = "home_position"
    TYPE          = "type"
    STATE_UPDATE  = "STATE_UPDATE"
