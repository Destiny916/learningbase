# 保留原有的工具函数，添加网络相关功能
from multiprocessing import Queue
import multiprocessing as mp
from dataclasses import dataclass
import logging
import os
import time
import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import deque
import threading
from act_async_infer_distributed_demo.scripts.w1_mapping import (
    ImageKey,
    InferenceConfigKey,
    CommonKey,
)
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from functools import wraps

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the default log level
logger.setLevel(logging.DEBUG)
def count_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)
    
    wrapper.call_count = 0
    def reset_counter():
        wrapper.call_count = 0
    
    wrapper.reset_counter = reset_counter
    return wrapper

def set_seed(seed: int = 42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def decorate_str_color(msg: str, color: str):
    """Decorate a string with a specific color."""
    color_map = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "orange": "\033[33m",
        "white": "\033[97m",
    }
    return f"{color_map.get(color, '')}{msg}\033[0m" if color else msg

def log_section(title="", symbol="="):
    """输出带分隔线的日志部分"""
    separator = symbol*60
    if title:
        log_info(f"\n{separator}\n {title} \n{separator}")
    else:
        log_info(f"\n{separator}")

def set_log_level(level: str):
    """Set the logging level."""
    level = level.upper()
    assert level in ["DEBUG", "INFO", "WARNING", "ERROR"], "Invalid log level"
    logger.setLevel(getattr(logging, level))


def format_message(level: str, message: str):
    """Format the log message with a consistent prefix."""
    return f"[DexforceVLA {level}]: {message}"


def log_info(message, color=None):
    """Log an info message."""
    logger.info(decorate_str_color(format_message("INFO", message), "green"))


def log_debug(message):
    """Log a debug message."""
    logger.debug(decorate_str_color(format_message("DEBUG", message), "blue"))
    return message

def log_warning(message):
    """Log a warning message."""
    logger.warning(decorate_str_color(format_message("WARNING", message), "purple"))
    return message


@count_calls
def log_error(message, error_type=RuntimeError):
    """Log an error message."""
    logger.error(decorate_str_color(format_message("ERROR", message), "red"))
    return message


def ensure_block(out: List[str], block: List[str]):
    # 若 6 个名字都未出现，则补上整块
    if not any(n in out for n in block):
        out.extend(block)


def save_vis_images(output_dir, timestep, vis_images):
    for j, image in enumerate(vis_images):
        cv2.imwrite(
            os.path.join(
                output_dir,
                "vis",
                "vis_{}_{}.png".format(timestep, j),
            ),
            image,
        )


def load_json(path: str) -> Dict:
    import json

    with open(path) as f:
        config = json.load(f)
    return config


import numpy as np


def find_model_dirs(root_path: str):
    """
    递归查找同时包含 model.safetensors和 config.json 的目录
    返回符合这个目录的当前路径列表
    
    Args:
        root_path: 起始搜索路径
        
    Returns:
        List[str]: 符合条件目录的当前路径列表（绝对路径）
    """
    results = []
    required_files = {"model.safetensors", "config.json"}
    
    if root_path is None or not os.path.exists(root_path):
        return results
    # 确保输入路径是绝对路径
    root_path = os.path.abspath(root_path)
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # 转换为集合以便快速查找
        file_set = set(filenames)
        
        # 检查是否同时包含三个必需文件
        if required_files.issubset(file_set):
            results.append(os.path.abspath(dirpath))

    return results

def filter_and_rearrange_proprio(proprio_array, origin_names, target_names):

    name_to_index = {name: idx for idx, name in enumerate(origin_names)}

    missing_names = [name for name in target_names if name not in name_to_index]
    if missing_names:
        raise ValueError(f"以下关节名称在原始列表中不存在: {missing_names}")

    target_indices = [name_to_index[name] for name in target_names]

    return proprio_array[target_indices]


def generate_cos_wave(
    action_chunk: dict,
):
    import torch
    from embodichain.data.enum import JointType, SUPPORTED_PROPRIO_TYPES

    if JointType.QPOS.value in action_chunk:
        qpos_dict = action_chunk[JointType.QPOS.value]

        # 获取序列长度
        if SUPPORTED_PROPRIO_TYPES[2] in qpos_dict:
            seq_len = len(qpos_dict[SUPPORTED_PROPRIO_TYPES[2]])
        elif SUPPORTED_PROPRIO_TYPES[3] in qpos_dict:
            seq_len = len(qpos_dict[SUPPORTED_PROPRIO_TYPES[3]])
        else:
            seq_len = 64  # 默认长度

        import numpy as np

        t = np.linspace(0, 2 * np.pi, seq_len)
        sine_wave = np.cos(t)

        # 替换左右手臂动作的第一个维度
        for key in [SUPPORTED_PROPRIO_TYPES[2], SUPPORTED_PROPRIO_TYPES[3]]:
            if key in qpos_dict:
                if isinstance(qpos_dict[key], torch.Tensor):
                    qpos_array = qpos_dict[key].cpu().numpy()
                else:
                    qpos_array = qpos_dict[key]

                # 替换第一个维度
                qpos_array[:, 0] = sine_wave

    return action_chunk


def get_kingfisher_r_image_info(kingfisher_image_info_queue: Queue):
    import kingfisher

    c = kingfisher.connect("192.168.1.188")
    import json

    left, right = kingfisher.captureQuarterSize()
    data = {
        ImageKey.CAM_HIGH_LEFT.value: {
            CommonKey.SHAPE.value: left.shape,
            CommonKey.SIZE.value: left.nbytes,
        },
        ImageKey.CAM_HIGH_RIGHT.value: {
            CommonKey.SHAPE.value: right.shape,
            CommonKey.SIZE.value: right.nbytes,
        },
    }
    kingfisher_image_info_queue.put(json.dumps(data))


def setup_logger():

    formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def compress_image_for_network(image, quality=80):
    """压缩图像用于网络传输"""
    if image is None:
        return None
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_image = cv2.imencode(".jpg", image, encode_param)
    if result:
        return encoded_image
    return None


def decompress_image_from_network(compressed_data):
    """从网络数据解压缩图像"""
    if compressed_data is None:
        return None
    return cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)


@dataclass
class TimedData:
    """A data object with timestamp and timestep information.

    Args:
        timestamp: Unix timestamp relative to data's creation.
        data: The actual data to wrap a timestamp around.
        timestep: The timestep of the data.
    """

    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: np.ndarray

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: dict
    must_go: bool = False

    def get_observation(self):
        return self.observation


@dataclass
class TimedFrame:
    t: float
    left_bgr: np.ndarray
    right_bgr: np.ndarray


def now_sec() -> float:
    import time

    return time.time()


def stamp_to_sec(stamp) -> float:
    if stamp and (stamp.sec or stamp.nanosec):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return now_sec()


def nearest(
    buf: deque, t: float, tol_s: float, lock: Optional[threading.Lock] = None
) -> Optional[np.ndarray]:
    """Return latest value in buf nearest to time t within tol."""
    if not buf:
        return None

    # 使用锁保护并发访问
    if lock:
        with lock:
            if not buf:  # 再次检查，因为可能在其他线程中清空
                return None
            buf_copy = list(buf)  # 创建副本
    else:
        buf_copy = list(buf)

    if not buf_copy:
        return None

    ts = np.fromiter((x[0] for x in buf_copy), dtype=np.float64)
    i = int(np.argmin(np.abs(ts - t)))
    dt = float(ts[i] - t)
    if abs(dt) <= tol_s:
        return buf_copy[i][1]
    return None

def nearest_without_tol(
    buf: deque, lock: Optional[threading.Lock] = None
) -> Optional[np.ndarray]:
    """对buf取最后一个动作(buf是deque,最新的元素在末位,每个元素是t和数据,所以数据是[-1][1])"""
    if not buf:
        return None
    else:
        with lock:
            return buf[-1][1]
    

def normalize_array(arr, src_min=0, src_max=1, dst_min=0, dst_max=100, reverse=False):

    # 防止除零错误
    if src_max == src_min:
        return arr if isinstance(arr, np.ndarray) else np.array(arr)
    
    # 转换为 numpy 数组
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=float)
    
    if reverse:
        # 反转映射
        result = (src_max - arr) * (dst_max - dst_min) / (src_max - src_min) + dst_min
    else:
        # 正常映射
        result = (arr - src_min) * (dst_max - dst_min) / (src_max - src_min) + dst_min
    
    return result



def draw_actionchunks_dt(
    target_qpos_list: List[TimedAction] = [],
    actionchunks: List = [],
    plot_path=None,
    inference_delays: List[float] = None,
    blending_zones: List[float] = None,
    connection_points: List[dict] = None,
    dt: float = 0.1,  # 期望的时间间隔，需要传入
    threshold_factor: float = 1.5,  # 阈值因子，用于判断是否"远大于"
):
    import matplotlib.pyplot as plt
    import matplotlib
    import copy

    matplotlib.use("Agg")

    # 深拷贝数据，以免修改原始数据
    compensated_target_qpos_list = copy.deepcopy(target_qpos_list)
    compensated_actionchunks = copy.deepcopy(actionchunks)
    compensated_connection_points = (
        copy.deepcopy(connection_points) if connection_points else None
    )

    # 只处理第一个和第二个连接点
    if compensated_connection_points and len(compensated_connection_points) >= 2:
        # 获取第一个和第二个连接点
        cp1 = compensated_connection_points[0]
        cp2 = compensated_connection_points[1]

        cp1_timestamp = cp1.get(CommonKey.TIMESTAMP.value)
        cp2_timestamp = cp2.get(CommonKey.TIMESTAMP.value)

        if cp1_timestamp is not None and cp2_timestamp is not None:
            # 查找第一个连接点之前的最后一个实际执行动作
            prev_action = None
            post_action = None
            for action in reversed(compensated_target_qpos_list):
                if action.get_timestamp() < cp2_timestamp:
                    prev_action = action
                if action.get_timestamp() >= cp2_timestamp:
                    post_action = action
                if prev_action is not None and post_action is not None:
                    break

            if prev_action is not None and post_action is not None:
                # 计算时间间隔K
                K = post_action.get_timestamp() - prev_action.get_timestamp()

                # 检查是否远大于dt
                if K > dt * threshold_factor:
                    compensation = K - dt

                    # 补偿第一个actionchunk中的所有动作
                    if len(compensated_actionchunks) > 0:
                        first_chunk = compensated_actionchunks[0]
                        for action_in_chunk in first_chunk:
                            action_in_chunk.timestamp = (
                                action_in_chunk.get_timestamp() + compensation
                            )

                    # 补偿两个连接点之间的所有实际执行动作
                    for action in compensated_target_qpos_list:
                        action_timestamp = action.get_timestamp()
                        if cp1_timestamp <= action_timestamp < cp2_timestamp:
                            action.timestamp = action_timestamp + compensation

                    compensated_connection_points[0][CommonKey.TIMESTAMP.value] = (
                        compensated_connection_points[0][CommonKey.TIMESTAMP.value]
                        + compensation
                    )
                    log_info(
                        f"Applied compensation of {compensation}s for interval K={K}s between CP1 and CP2"
                    )

    # 使用补偿后的数据进行绘图
    ref_qpos_y = [x.get_action() for x in compensated_target_qpos_list if x is not None]
    ref_qpos_y = np.array(ref_qpos_y)
    first_timestamp = compensated_target_qpos_list[0].get_timestamp()
    ref_qpos_x = [
        x.get_timestamp() - first_timestamp
        for x in compensated_target_qpos_list
        if x is not None
    ]
    ref_qpos_x = np.array(ref_qpos_x)

    actionchunks_timestamps = []
    actionchunks_values = []

    for actionchunk in compensated_actionchunks:
        chunk_timestamps = [a.get_timestamp() - first_timestamp for a in actionchunk]
        chunk_values = [a.get_action() for a in actionchunk]

        actionchunks_timestamps.append(np.array(chunk_timestamps))
        actionchunks_values.append(np.array(chunk_values))

    connection_points_x = []
    connection_points_y = []
    if compensated_connection_points:
        for cp in compensated_connection_points:
            cp_timestamp = cp.get(CommonKey.TIMESTAMP.value)
            if cp_timestamp:
                cp_rel_time = cp_timestamp - first_timestamp
                connection_points_x.append(cp_rel_time)
                cp_action = cp.get(CommonKey.ACTION.value)
                if cp_action is not None:
                    connection_points_y.append(cp_action)

    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(compensated_actionchunks))]
    labels = [f"Chunk #{i}" for i in range(len(compensated_actionchunks))]

    num_ts, num_dim = ref_qpos_y.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(32, 4 * num_dim))

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]

        ax.plot(
            ref_qpos_x,
            ref_qpos_y[:, dim_idx],
            color="black",
            label="Executed Actions",
            marker="o",
            markersize=3,
            linewidth=2,
            zorder=10,
        )

        for i, (chunk_ts, chunk_vals, color, label) in enumerate(
            zip(actionchunks_timestamps, actionchunks_values, colors, labels)
        ):
            if dim_idx < chunk_vals.shape[1]:
                ax.plot(
                    chunk_ts,
                    chunk_vals[:, dim_idx],
                    color=color,
                    label=label,
                    marker="o",
                    markersize=2,
                    linewidth=1,
                    linestyle="--",
                    alpha=0.7,
                    zorder=5,
                )
        # 绘制推理延迟和融合区域 - 从chunk开始，跳过第一个chunk
        if inference_delays is not None and blending_zones is not None:
            # 从第二个chunk开始绘制（索引1）
            for i in range(1, len(compensated_actionchunks)):
                # 确保有对应的inference_delay和blending_zone
                if i >= len(inference_delays) or i >= len(blending_zones):
                    continue

                # 获取当前chunk的开始时间
                if len(actionchunks_timestamps[i]) > 0:
                    chunk_start = actionchunks_timestamps[i][0]
                else:
                    continue

                delay = inference_delays[i]
                blend = blending_zones[i]

                # 绘制inference delay（从chunk开始）
                if delay > 0:
                    delay_start = chunk_start
                    delay_end = chunk_start + delay

                    if i == 1 and dim_idx == 0:  # 第一个绘制的区域（i=1）
                        ax.axvspan(
                            delay_start,
                            delay_end,
                            color="gray",
                            alpha=0.2,
                            label="Inference Delay",
                        )
                    else:
                        ax.axvspan(delay_start, delay_end, color="gray", alpha=0.2)

                # 绘制blending zone（紧跟在inference delay后面）
                if blend > 0:
                    blend_start = chunk_start + delay
                    blend_end = chunk_start + delay + blend

                    if i == 1 and dim_idx == 0:  # 第一个绘制的区域（i=1）
                        ax.axvspan(
                            blend_start,
                            blend_end,
                            color="yellow",
                            alpha=0.2,
                            label="Blending Zone",
                        )
                    else:
                        ax.axvspan(blend_start, blend_end, color="yellow", alpha=0.2)

                # 在chunk开始处画垂直线
                if i == 1 and dim_idx == 0:
                    ax.axvline(
                        x=chunk_start,
                        color="gray",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.5,
                        zorder=4,
                        label="Chunk Start",
                    )
                else:
                    ax.axvline(
                        x=chunk_start,
                        color="gray",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.5,
                        zorder=4,
                    )

        # 绘制连接点标记
        if connection_points_x:
            cp_y_values = []
            for cp_y in connection_points_y:
                if dim_idx < len(cp_y):
                    cp_y_values.append(cp_y[dim_idx])
                else:
                    cp_y_values.append(0)

            ax.scatter(
                connection_points_x,
                cp_y_values,
                marker="*",
                s=100,
                color="red",
                zorder=15,
                label="Connection Point" if dim_idx == 0 else "",
            )

            # 在连接点处画垂直线
            for cp_x in connection_points_x:
                ax.axvline(
                    x=cp_x,
                    color="red",
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.3,
                    zorder=3,
                )

        ax.set_title(f"Joint {dim_idx} (rad)")
        ax.set_xlabel("Relative Time (s)")
        ax.set_ylabel("Joint Angle (rad)")
        ax.grid(True, alpha=0.3)

        if dim_idx == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label="Executed Actions",
                ),
            ]

            max_chunk_legend = min(3, len(colors))
            for i in range(max_chunk_legend):
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color=colors[i],
                        marker="o",
                        markersize=1,
                        linewidth=0.5,
                        linestyle="--",
                        alpha=0.7,
                        label=f"Actionchunk #{i}",
                    )
                )

            if inference_delays is not None:
                legend_elements.append(
                    Patch(facecolor="gray", alpha=0.2, label="Inference Delay")
                )

            if blending_zones is not None:
                legend_elements.append(
                    Patch(facecolor="yellow", alpha=0.2, label="Blending Zone")
                )

            if connection_points_x:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="*",
                        color="w",
                        markerfacecolor="red",
                        markersize=10,
                        label="Connection Point",
                    )
                )

            # 添加chunk开始线的图例
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.5,
                    label="Chunk Start",
                )
            )

            if len(colors) > max_chunk_legend:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        linestyle="--",
                        linewidth=1,
                        label=f"... and {len(colors) - max_chunk_legend} more chunks",
                    )
                )

            ax.legend(handles=legend_elements, loc="upper left", fontsize="small")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    log_info(f"plot save to: {plot_path}")
    log_info(f"plot total {len(compensated_actionchunks)} actionchunks")
    log_info(f"plot mark {len(connection_points_x)} connect points")
    if inference_delays is not None:
        log_info(f"Inference Delays: {inference_delays}")
    if blending_zones is not None:
        log_info(f"Blending Zones: {blending_zones}")


def draw_actionchunks(
    target_qpos_list: List[TimedAction] = [],
    actionchunks: List = [],
    plot_path=None,
    inference_delays: List[float] = None,
    blending_zones: List[float] = None,
    connection_points: List[dict] = None,
):
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.set_loglevel("warning")
    ref_qpos_y = [x.get_action() for x in target_qpos_list if x is not None]
    ref_qpos_y = np.array(ref_qpos_y)
    first_timestamp = target_qpos_list[0].get_timestamp()
    ref_qpos_x = [
        x.get_timestamp() - first_timestamp for x in target_qpos_list if x is not None
    ]
    ref_qpos_x = np.array(ref_qpos_x)

    actionchunks_timestamps = []
    actionchunks_values = []

    for actionchunk in actionchunks:
        chunk_timestamps = [a.get_timestamp() - first_timestamp for a in actionchunk]
        chunk_values = [a.get_action() for a in actionchunk]

        actionchunks_timestamps.append(np.array(chunk_timestamps))
        actionchunks_values.append(np.array(chunk_values))

    connection_points_x = []
    connection_points_y = []
    if connection_points:
        for cp in connection_points:
            cp_timestamp = cp.get(CommonKey.TIMESTAMP.value)
            if cp_timestamp:
                cp_rel_time = cp_timestamp - first_timestamp
                connection_points_x.append(cp_rel_time)
                cp_action = cp.get(CommonKey.ACTION.value)
                if cp_action is not None:
                    connection_points_y.append(cp_action)

    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(actionchunks))]
    labels = [f"Chunk #{i}" for i in range(len(actionchunks))]

    num_ts, num_dim = ref_qpos_y.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(32, 4 * num_dim))

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]

        ax.plot(
            ref_qpos_x,
            ref_qpos_y[:, dim_idx],
            color="black",
            label="Executed Actions",
            marker="o",
            markersize=3,
            linewidth=2,
            zorder=10,
        )

        for i, (chunk_ts, chunk_vals, color, label) in enumerate(
            zip(actionchunks_timestamps, actionchunks_values, colors, labels)
        ):
            if dim_idx < chunk_vals.shape[1]:
                ax.plot(
                    chunk_ts,
                    chunk_vals[:, dim_idx],
                    color=color,
                    label=label,
                    marker="o",
                    markersize=2,
                    linewidth=1,
                    linestyle="--",
                    alpha=0.7,
                    zorder=5,
                )

        # 绘制推理延迟和融合区域 - 从chunk开始，跳过第一个chunk
        if inference_delays is not None and blending_zones is not None:
            # 从第二个chunk开始绘制（索引1）
            for i in range(1, len(actionchunks)):
                # 确保有对应的inference_delay和blending_zone
                if i >= len(inference_delays) or i >= len(blending_zones):
                    continue

                # 获取当前chunk的开始时间
                if len(actionchunks_timestamps[i]) > 0:
                    chunk_start = actionchunks_timestamps[i][0]
                else:
                    continue

                delay = inference_delays[i]
                blend = blending_zones[i]

                # 绘制inference delay（从chunk开始）
                if delay > 0:
                    delay_start = chunk_start
                    delay_end = chunk_start + delay

                    if i == 1 and dim_idx == 0:  # 第一个绘制的区域（i=1）
                        ax.axvspan(
                            delay_start,
                            delay_end,
                            color="gray",
                            alpha=0.2,
                            label="Inference Delay",
                        )
                    else:
                        ax.axvspan(delay_start, delay_end, color="gray", alpha=0.2)

                # 绘制blending zone（紧跟在inference delay后面）
                if blend > 0:
                    blend_start = chunk_start + delay
                    blend_end = chunk_start + delay + blend

                    if i == 1 and dim_idx == 0:  # 第一个绘制的区域（i=1）
                        ax.axvspan(
                            blend_start,
                            blend_end,
                            color="yellow",
                            alpha=0.2,
                            label="Blending Zone",
                        )
                    else:
                        ax.axvspan(blend_start, blend_end, color="yellow", alpha=0.2)

                # 在chunk开始处画垂直线
                if i == 1 and dim_idx == 0:
                    ax.axvline(
                        x=chunk_start,
                        color="gray",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.5,
                        zorder=4,
                        label="Chunk Start",
                    )
                else:
                    ax.axvline(
                        x=chunk_start,
                        color="gray",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.5,
                        zorder=4,
                    )

        # 绘制连接点标记
        if connection_points_x:
            cp_y_values = []
            for cp_y in connection_points_y:
                if dim_idx < len(cp_y):
                    cp_y_values.append(cp_y[dim_idx])
                else:
                    cp_y_values.append(0)

            ax.scatter(
                connection_points_x,
                cp_y_values,
                marker="*",
                s=100,
                color="red",
                zorder=15,
                label="Connection Point" if dim_idx == 0 else "",
            )

            # 在连接点处画垂直线
            for cp_x in connection_points_x:
                ax.axvline(
                    x=cp_x,
                    color="red",
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.3,
                    zorder=3,
                )

        ax.set_title(f"Joint {dim_idx} (rad)")
        ax.set_xlabel("Relative Time (s)")
        ax.set_ylabel("Joint Angle (rad)")
        ax.grid(True, alpha=0.3)

        if dim_idx == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    label="Executed Actions",
                ),
            ]

            max_chunk_legend = min(3, len(colors))
            for i in range(max_chunk_legend):
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color=colors[i],
                        marker="o",
                        markersize=1,
                        linewidth=0.5,
                        linestyle="--",
                        alpha=0.7,
                        label=f"Actionchunk #{i}",
                    )
                )

            if inference_delays is not None:
                legend_elements.append(
                    Patch(facecolor="gray", alpha=0.2, label="Inference Delay")
                )

            if blending_zones is not None:
                legend_elements.append(
                    Patch(facecolor="yellow", alpha=0.2, label="Blending Zone")
                )

            if connection_points_x:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="*",
                        color="w",
                        markerfacecolor="red",
                        markersize=10,
                        label="Connection Point",
                    )
                )

            # 添加chunk开始线的图例
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.5,
                    label="Chunk Start",
                )
            )

            if len(colors) > max_chunk_legend:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        linestyle="--",
                        linewidth=1,
                        label=f"... and {len(colors) - max_chunk_legend} more chunks",
                    )
                )

            ax.legend(handles=legend_elements, loc="upper left", fontsize="small")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    log_info(f"plot save to: {plot_path}")
    log_info(f"plot total {len(actionchunks)}  actionchunks")
    log_info(f"plot mark {len(connection_points_x)} connect points")
    if inference_delays is not None:
        log_info(f"Inference Delays: {inference_delays}")
    if blending_zones is not None:
        log_info(f"Blending Zones: {blending_zones}")


def _save_frames_to_file(recorded_frames, record_filename):
    import json

    try:
        action_data = {"language_prompt": " ", "frames": recorded_frames}
        with open(record_filename, "w") as f:
            json.dump(action_data, f, indent=2)
        log_info(f"动作已保存到文件: {record_filename}")
    except Exception as e:
        log_error(f"保存动作文件失败: {e}")


def interpolate_2d(actions, factor):

    original_length, n_columns = actions.shape
    new_length = int(original_length * factor)
    new_length = max(1, new_length)

    if original_length == new_length:
        return actions.copy()

    if original_length == 1:
        return np.tile(actions, (new_length, 1))

    t_original = np.linspace(0, 1, original_length)
    t_target = np.linspace(0, 1, new_length)

    f_interp = interp1d(
        t_original,
        actions,
        axis=0,
        kind="linear",
        fill_value=(actions[0, :], actions[-1, :]),
        bounds_error=False,
    )

    return f_interp(t_target)


def interpolate_1d(timestamp, factor):

    from scipy.interpolate import interp1d

    n_original = len(timestamp)
    n_target = int(n_original * factor)

    if n_original <= 1 or n_target == n_original:
        return timestamp.copy()

    x_original = np.arange(n_original)

    x_target = np.linspace(0, n_original - 1, n_target)

    interp_func = interp1d(
        x_original,
        timestamp,
        kind="linear",
        fill_value=(timestamp[0], timestamp[-1]),
        bounds_error=False,
    )

    interpolated = interp_func(x_target)
    return interpolated


class Debug_Datasaver:
    def __init__(
        self,
        base_output_dir="output",
        save_input=False,
        save_vis_image=False,
        save_joint_change=False,
        camera_used=None,
    ):
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"{base_output_dir}/debug_{self.timestamp}"
        self.camera_used = camera_used or []
        self.count = 0
        self.save_input = save_input
        self.save_joint_change = save_joint_change
        os.makedirs(self.base_dir, exist_ok=True)
        log_info(f"Debug_Datasaver initialized. Data will be saved to: {self.base_dir}")

    def save_batch(self, images, states_value, prompt_value, additional_info=None):
        batch_dir = f"{self.base_dir}/batch_{self.count:04d}"
        input_batch_dir = f"{batch_dir}/input_batch"

        os.makedirs(input_batch_dir, exist_ok=True)

        self._save_images(images, input_batch_dir)

        self._save_states(states_value, input_batch_dir)
        
        self._save_prompt(prompt_value, input_batch_dir)
        
        self._save_batch_info(input_batch_dir, additional_info)
        
        if additional_info and "action_chunk" in additional_info:
            import pickle
            pkl_path = f"{batch_dir}/action_chunk.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(additional_info["action_chunk"], f)
            log_info(f"Saved action_chunk pickle to: {pkl_path}")
        
        log_info(f"Saved batch {self.count} to: {batch_dir}")
        return batch_dir


    def _save_images(self, images, batch_dir):
        for cam_idx in range(len(self.camera_used)):
            if cam_idx < len(images):
                img = images[cam_idx]
                cv2.imwrite(f"{batch_dir}/camera_{cam_idx:02d}.png", img)

    def _save_states(self, states_value, batch_dir):
        with open(f"{batch_dir}/states.txt", "w") as file:
            file.write(str(states_value))

    def _save_prompt(self, prompt_value, batch_dir):
        with open(f"{batch_dir}/prompt.txt", "w") as file:
            prompt_dict = {"prompt": prompt_value}
            file.write(str(prompt_dict))

    def _save_action_chunk_pkl(self, action_chunk):
        import pickle
        batch_dir = f"{self.base_dir}/batch_{self.count:04d}"
        pkl_path = f"{batch_dir}/action_chunk.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(action_chunk, f)
        log_info(f"Saved action_chunk pickle to: {pkl_path}")

    def _save_batch_info(self, batch_dir, additional_info):
        with open(f"{batch_dir}/batch_info.txt", "w") as file:
            file.write(f"Batch ID: {self.count}\n")
            file.write(f"Number of cameras: {len(self.camera_used)}\n")
            file.write(f"Save time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if additional_info:
                file.write(f"Additional info: {additional_info}\n")

    def update_save_path(self):
        self.count += 1

    def _save_vis_images(self, vis_images):
        batch_dir = f"{self.base_dir}/batch_{self.count:04d}"
        vis_image_dir = f"{batch_dir}/vis_image"
        os.makedirs(vis_image_dir, exist_ok=True)
        if vis_images:
            for j, image in enumerate(vis_images):
                cv2.imwrite(
                    os.path.join(vis_image_dir, "vis_{}.png".format(j)),
                    image,
                )

    def get_save_dir(self):
        return self.base_dir

    def _plot_joint_subplots(
        self,
        data: np.ndarray,
        labels: list,
        title: str,
        save_path: str,
        color: str = "b-",
        show_range: bool = False,
    ):
        """
        通用关节子图绘制。

        Args:
            data: shape (time_steps, num_joints)
            labels: 每个子图的 y 轴标签，长度 = num_joints
            title: 总标题
            save_path: 保存路径
            color: 线条颜色样式
            show_range: 是否显示 min/max
        """
        import matplotlib.pyplot as plt

        num_joints = data.shape[1]
        time_steps = np.arange(data.shape[0])

        fig, axes = plt.subplots(num_joints, 1, figsize=(15, 3 * num_joints))
        fig.suptitle(title, fontsize=16)

        # 单子图时 axes 不是数组，统一处理
        if num_joints == 1:
            axes = [axes]

        for i in range(num_joints):
            ax = axes[i]
            ax.plot(time_steps, data[:, i], color, linewidth=2)
            ax.set_ylabel(labels[i], fontsize=10)
            ax.grid(True, alpha=0.3)

            mean_val = np.mean(data[:, i])
            std_val = np.std(data[:, i])

            if show_range:
                min_val = np.min(data[:, i])
                max_val = np.max(data[:, i])
                stats_text = (
                    f"Mean: {mean_val:.4f}\n"
                    f"Std:  {std_val:.4f}\n"
                    f"Range: [{min_val:.4f}, {max_val:.4f}]"
                )
            else:
                stats_text = f"Mean: {mean_val:.4f}\nStd:  {std_val:.4f}"

            ax.text(
                0.02, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


    def _plot_joint_comparison_subplots(
        self,
        old_data: np.ndarray,
        new_data: np.ndarray,
        labels: list,
        title: str,
        save_path: str,
        old_label: str = "Previous",
        new_label: str = "Current",
        show_range: bool = False,
    ):
        import matplotlib.pyplot as plt

        num_joints = old_data.shape[1]
        time_steps = np.arange(old_data.shape[0])

        fig, axes = plt.subplots(num_joints, 1, figsize=(15, 3 * num_joints))
        fig.suptitle(title, fontsize=16)

        if num_joints == 1:
            axes = [axes]

        for i in range(num_joints):
            ax = axes[i]
            ax.plot(time_steps, old_data[:, i], "b--", linewidth=1.5, alpha=0.6, label=old_label)
            ax.plot(time_steps, new_data[:, i], "r-", linewidth=2, label=new_label)
            ax.set_ylabel(labels[i], fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            for color, data in [("blue", old_data), ("red", new_data)]:
                mean_val = np.mean(data[:, i])
                std_val = np.std(data[:, i])
                if show_range:
                    min_val, max_val = np.min(data[:, i]), np.max(data[:, i])
                    stats = f"{color}: μ={mean_val:.3f} σ={std_val:.3f} [{min_val:.3f},{max_val:.3f}]"
                else:
                    stats = f"{color}: μ={mean_val:.3f} σ={std_val:.3f}"

            ax.text(
                0.02, 0.95, f"New μ={np.mean(new_data[:,i]):.3f}\nOld μ={np.mean(old_data[:,i]):.3f}",
                transform=ax.transAxes, fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


    def plot_joint_changes(self, action_chunk_data, prev_pickle_path=None):
        batch_dir = f"{self.base_dir}/batch_{self.count:04d}"
        self.save_dir = batch_dir
        output_joint_dir = f"{batch_dir}/joint_output"
        os.makedirs(output_joint_dir, exist_ok=True)

        # 加载历史对比数据
        prev_data = None
        if prev_pickle_path:
            import pickle
            try:
                with open(prev_pickle_path, "rb") as f:
                    prev_data = pickle.load(f)
                log_info(f"Loaded previous pickle for comparison: {prev_pickle_path}")
            except Exception as e:
                log_warning(f"Failed to load previous pickle: {e}")

        
        from act_async_infer_distributed_demo.scripts.w1_mapping import w1qpos_names_map
        for config_key, labels, name, show_range in [
            (InferenceConfigKey.LEFT_ARMQPOS.value, w1qpos_names_map[InferenceConfigKey.LEFT_ARMQPOS.value], InferenceConfigKey.LEFT_ARMQPOS.value, True),
            (InferenceConfigKey.RIGHT_ARMQPOS.value, w1qpos_names_map[InferenceConfigKey.RIGHT_ARMQPOS.value], InferenceConfigKey.RIGHT_ARMQPOS.value, True),
            (InferenceConfigKey.LEFT_EEFHAND.value, w1qpos_names_map[InferenceConfigKey.LEFT_EEFHAND.value], InferenceConfigKey.LEFT_EEFHAND.value, False),
            (InferenceConfigKey.RIGHT_EEFHAND.value, w1qpos_names_map[InferenceConfigKey.RIGHT_EEFHAND.value], InferenceConfigKey.RIGHT_EEFHAND.value, False),
            (InferenceConfigKey.LEFT_EEFGRIPPER.value, w1qpos_names_map[InferenceConfigKey.LEFT_EEFGRIPPER.value], InferenceConfigKey.LEFT_EEFGRIPPER.value, False),
            (InferenceConfigKey.RIGHT_EEFGRIPPER.value, w1qpos_names_map[InferenceConfigKey.RIGHT_EEFGRIPPER.value], InferenceConfigKey.RIGHT_EEFGRIPPER.value, False),
        ]:
            new_val = action_chunk_data.get(config_key)
            old_val = prev_data.get(config_key) if prev_data else None

            if new_val is None:
                continue

            if old_val is not None and old_val.shape == new_val.shape:
                # 画对比图
                self._plot_joint_comparison_subplots(
                    old_data=old_val,
                    new_data=new_val,
                    labels=labels[:new_val.shape[1]],
                    title=f"{name} - Comparison (Previous vs Current)",
                    save_path=f"{output_joint_dir}/{name}_comparison.png",
                    show_range=show_range,
                )
            else:
                # 单独画
                self._plot_joint_subplots(
                    data=new_val,
                    labels=labels[:new_val.shape[1]],
                    title=f"{name}",
                    save_path=f"{output_joint_dir}/{name}.png",
                    color="r-",
                    show_range=show_range,
                )

        log_info(f"All joint plots saved to directory: {output_joint_dir}/")



def extract_actions_for_body(
    config_key, pub_names, pub_pos, config_indices, sim_init_pose
):
    indices = config_indices[config_key]
    actions = []
    # 根据config_key确定应该提取哪些关节
    if config_key == InferenceConfigKey.TORSO.value:
        # TORSO: ANKLE, KNEE, BUTTOCK, WAIST
        for i, idx in enumerate(indices):
            if idx != -1:
                actions.append(pub_pos[idx])
            else:
                # 使用sim_init_pose中对应的值
                init_idx = i
                if init_idx < len(sim_init_pose):
                    actions.append(sim_init_pose[init_idx])
                else:
                    actions.append(0.0)

    elif config_key == InferenceConfigKey.HEAD.value:
        # HEAD: NECK1, NECK2
        for i, idx in enumerate(indices):
            if idx != -1:
                actions.append(pub_pos[idx])
            else:
                init_idx = 4 + i
                if init_idx < len(sim_init_pose):
                    actions.append(sim_init_pose[init_idx])
                else:
                    actions.append(0.0)

    elif config_key == InferenceConfigKey.LEFT_ARMQPOS.value:
        # LEFT_ARM: LEFT_J1到LEFT_J7
        for i, idx in enumerate(indices):
            if idx != -1:
                actions.append(pub_pos[idx])
            else:
                init_idx = 6 + i
                if init_idx < len(sim_init_pose):
                    actions.append(sim_init_pose[init_idx])
                else:
                    actions.append(0.0)

    elif config_key == InferenceConfigKey.RIGHT_ARMQPOS.value:
        # RIGHT_ARM: RIGHT_J1到RIGHT_J7
        for i, idx in enumerate(indices):
            if idx != -1:
                actions.append(pub_pos[idx])
            else:
                init_idx = 13 + i
                if init_idx < len(sim_init_pose):
                    actions.append(sim_init_pose[init_idx])
                else:
                    actions.append(0.0)

    return actions

class RealTimeFrequencyMonitor:
        
    def __init__(self, window_size=60):

        self.window_size = window_size
        self.call_times = deque()
        self.lock = threading.Lock()
    
    def record_call(self):
        with self.lock:
            current_time = time.time()
            self.call_times.append(current_time)
            
            # 清理过期的调用记录
            while self.call_times and self.call_times[0] < current_time - self.window_size:
                self.call_times.popleft()
    
    def get_current_frequency(self):
        with self.lock:
            if not self.call_times:
                return 0.0
            
            current_time = time.time()
            window_start = current_time - self.window_size
            
            while self.call_times and self.call_times[0] < window_start:
                self.call_times.popleft()
            
            if not self.call_times:
                return 0.0
            
            time_span = current_time - self.call_times[0]
            if time_span == 0:
                return len(self.call_times)
            
            return len(self.call_times) / time_span
    
    def get_stats(self):
        with self.lock:
            current_time = time.time()
            window_start = current_time - self.window_size
            
            while self.call_times and self.call_times[0] < window_start:
                self.call_times.popleft()
            
            calls_in_window = len(self.call_times)
            
            if calls_in_window == 0:
                return {
                    'calls_in_window': 0,
                    'frequency': 0.0,
                    'window_size': self.window_size
                }
            
            actual_window = current_time - self.call_times[0]
            frequency = calls_in_window / actual_window if actual_window > 0 else calls_in_window
            
            return {
                'calls_in_window': calls_in_window,
                'frequency': frequency,
                'window_size': self.window_size,
                'actual_window': actual_window
            }
