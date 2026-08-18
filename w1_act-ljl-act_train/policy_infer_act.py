#!/usr/bin/env python3

import argparse
import gc
import json
import time
from multiprocessing import resource_tracker, shared_memory
from multiprocessing.connection import Listener

import numpy as np
import torch

# 【严格遵循】：使用原汁原味的 lerobot 框架
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION


def wall_ts_ns() -> str:
    t_ns = time.time_ns()
    sec, rem_ns = divmod(t_ns, 1_000_000_000)
    return time.strftime("%H:%M:%S", time.localtime(sec)) + f".{rem_ns:09d}"


def _read_obs_from_shm(shm_obs, num_slots, slot_size, state_dim, image_keys, state_key, img_shapes):
    """从共享内存零拷贝读取观测数据，重建为 dict 格式"""
    obs_np = {}
    for i, key in enumerate(image_keys):
        shape = img_shapes[key]
        byte_len = shape[0] * shape[1] * shape[2]
        offset = i * slot_size
        obs_np[key] = np.ndarray(shape, dtype=np.uint8, buffer=shm_obs.buf[offset : offset + byte_len]).copy()
    state_offset = num_slots * slot_size
    obs_np[state_key] = np.ndarray(
        (state_dim,), dtype=np.float32, buffer=shm_obs.buf[state_offset : state_offset + state_dim * 4]
    ).copy()
    return obs_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to server_config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    port = cfg.get("port", 8888)
    device_str = cfg.get("device", "cuda:0")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 {device_str}，但 CUDA 当前不可用；拒绝静默回退 CPU")
    device = torch.device(device_str)

    models_dict = {}
    print(f"[Model Server] 开始批量加载 Lerobot 模型字典到 {device} ...")

    for model_id, policy_path in cfg.get("models", {}).items():
        print(f"  -> 正在加载模型 [{model_id}]: {policy_path}")

        # 1:1 复刻 multi_process 加载逻辑
        train_config = TrainPipelineConfig.from_pretrained(policy_path, local_files_only=True)
        policy_config = train_config.policy
        if policy_config is None:
            raise ValueError(f"训练配置缺少 policy: {policy_path}")
        policy_config.device = str(device)
        policy = ACTPolicy.from_pretrained(
            policy_path,
            config=policy_config,
            local_files_only=True,
            strict=True,
        )
        policy.to(device).eval()

        if hasattr(policy, "config"):
            p_cfg = policy.config
            if hasattr(p_cfg, "temporal_ensemble_coeff"):
                p_cfg.temporal_ensemble_coeff = None

        try:
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                policy_path, config_filename="policy_preprocessor.json", local_files_only=True
            )
            postprocessor = PolicyProcessorPipeline.from_pretrained(
                policy_path, config_filename="policy_postprocessor.json", local_files_only=True
            )
        except Exception as exc:
            raise RuntimeError(f"模型 {model_id} 的前后处理器加载失败，拒绝未缩放推理") from exc
        for step in getattr(preprocessor, "steps", []):
            if step.__class__.__name__ == "DeviceProcessorStep":
                step.device = str(device)
            elif step.__class__.__name__ == "NormalizerProcessorStep" and hasattr(step, "to"):
                step.to(device=device)
        for step in getattr(postprocessor, "steps", []):
            if step.__class__.__name__ == "DeviceProcessorStep":
                step.device = "cpu"
            elif step.__class__.__name__ == "UnnormalizerProcessorStep" and hasattr(step, "to"):
                step.to(device="cpu")

        policy.reset()
        models_dict[model_id] = {"policy": policy, "pre": preprocessor, "post": postprocessor}
        print(f"  -> [{model_id}] 加载就绪。")

    if not models_dict:
        raise RuntimeError("配置文件中没有找到任何模型！")

    active_model_id = list(models_dict.keys())[0]

    # 共享内存状态（由 SHM_INIT 命令初始化）
    shm_obs = None
    shm_acts = None
    shm_num_slots = 0
    shm_slot_size = 0
    shm_state_dim = 0
    shm_horizon_n = 30
    shm_image_keys = []
    shm_state_key = ""
    shm_img_shapes = {}

    address = ("127.0.0.1", port)
    listener = None
    for _ in range(10):
        try:
            listener = Listener(address, authkey=b"w1_act_secret")
            break
        except OSError:
            print(f"[Model Server] 端口 {port} 被占用，等待释放中...")
            time.sleep(1)

    if not listener:
        raise RuntimeError(f"致命错误: 无法绑定端口 {port}")

    print(f"[Model Server] 引擎就绪！持续监听端口 {port}，当前激活 [{active_model_id}]")

    while True:
        try:
            conn = listener.accept()
            while True:
                try:
                    msg = conn.recv()
                    cmd = msg.get("cmd")

                    if cmd == "SHM_INIT":
                        shm_obs = shared_memory.SharedMemory(name=msg["obs_name"])
                        shm_acts = shared_memory.SharedMemory(name=msg["acts_name"])
                        resource_tracker.unregister(shm_obs._name, "shared_memory")
                        resource_tracker.unregister(shm_acts._name, "shared_memory")
                        shm_num_slots = msg["num_slots"]
                        shm_slot_size = msg["slot_size"]
                        shm_state_dim = msg["state_dim"]
                        shm_horizon_n = msg["horizon_N"]
                        shm_image_keys = msg["image_keys"]
                        shm_state_key = msg["state_key"]
                        shm_img_shapes = {key: tuple(msg["image_shapes"][key]) for key in shm_image_keys}
                        print(
                            f"[Model Server] 共享内存已连接 obs={msg['obs_size']}B acts={msg['acts_size']}B"
                        )
                        conn.send("OK")
                        continue

                    if cmd == "RESET":
                        models_dict[active_model_id]["policy"].reset()
                        conn.send("OK")
                        continue

                    if cmd == "SWITCH_MODEL":
                        target_id = msg.get("target")
                        if target_id in models_dict:
                            active_model_id = target_id
                            models_dict[active_model_id]["policy"].reset()

                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            conn.send("OK")
                        else:
                            conn.send("ERROR")
                        continue

                    if cmd == "INFER_CHUNK":
                        ts = wall_ts_ns()
                        active = models_dict[active_model_id]
                        policy, preprocessor, postprocessor = active["policy"], active["pre"], active["post"]

                        obs_np = _read_obs_from_shm(
                            shm_obs,
                            shm_num_slots,
                            shm_slot_size,
                            shm_state_dim,
                            shm_image_keys,
                            shm_state_key,
                            shm_img_shapes,
                        )
                        steps = msg.get("steps", 30)

                        obs_t = prepare_observation_for_inference(obs_np, device)
                        if preprocessor:
                            batch_torch = preprocessor(obs_t)
                        else:
                            batch_torch = {
                                k: v.unsqueeze(0) if v.ndim in (1, 3) else v for k, v in obs_t.items()
                            }
                        t_infer_start = time.perf_counter()
                        with torch.no_grad():
                            full_chunk = policy.predict_action_chunk(batch_torch)
                            if not 1 <= steps <= full_chunk.shape[1] or steps > shm_horizon_n:
                                raise ValueError(
                                    f"请求执行 {steps} 步，但模型预测长度为 {full_chunk.shape[1]}，"
                                    f"共享内存容量为 {shm_horizon_n}"
                                )
                            selected = full_chunk[:, :steps]
                            out = postprocessor({ACTION: selected})
                            actions_np = out[ACTION].detach().cpu().numpy()[0]
                        t_infer_end = time.perf_counter()

                        acts_v = np.ndarray(
                            (shm_horizon_n, shm_state_dim), dtype=np.float32, buffer=shm_acts.buf
                        )
                        acts_v[:steps] = actions_np
                        conn.send({"status": "OK", "n_steps": steps})

                        infer_ms = (t_infer_end - t_infer_start) * 1000
                        print(
                            f"[Model Server][{ts}] model={active_model_id} steps={steps} "
                            f"推理={infer_ms:.2f}ms"
                        )

                except EOFError:
                    models_dict[active_model_id]["policy"].reset()
                    conn.close()
                    break
        except Exception as e:
            print(f"[Model Server] 监听异常: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
