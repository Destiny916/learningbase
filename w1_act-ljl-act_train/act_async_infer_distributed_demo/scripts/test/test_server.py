import argparse
import json
import time
import threading
import numpy as np
import cv2
import pickle
import os
from copy import deepcopy
from queue import Queue, Empty
import sys


from act_async_infer_distributed_demo.scripts.w1_mapping import CommonKey, ImageKey, InferenceConfigKey, w1qpos_names_map
from act_async_infer_distributed_demo.scripts.utils_distributed import TimedObservation, log_info, log_error, log_warning
from act_async_infer_distributed_demo.scripts.network_utils import NetworkClient, compress_image


class AsyncMockRobotClient:
    def __init__(self, args, model_config, server_host, server_port):
        self.args = args
        self.model_config = model_config
        self.server_host = server_host
        self.server_port = server_port
        self.current_step = 0
        self.max_steps = 10

        self.head_target_size = model_config.get("head_target_size", [640, 360])
        self.hand_target_size = model_config.get("hand_target_size", [640, 480])

        self.network_client = NetworkClient(server_host, server_port)

        self.observation_data = []
        self.action_responses = []

        self.shutdown_event = threading.Event()
        self.connected = False
        self.allow_infer_check = True  # 是否允许检查推理条件
        self.start_infer = False  # 是否触发推理

        self.observation_queue = Queue(maxsize=1)
        self.action_queue = Queue()
        self.actions_ready = threading.Event()

        self.action_chunk_size = 0
        self._chunk_size_threshold = 0.5  # 队列阈值

        self.start_barrier = threading.Barrier(4)

        # 统计信息
        self.stats = {
            "observations_sent": 0,
            "actions_received": 0,
            "inferences_triggered": 0,
            "avg_observation_latency": 0,
            "avg_action_latency": 0,
        }

        self.joint_limits = {
            InferenceConfigKey.WAISTQPOS.value: (-1.0, 1.0),  # 腰部
            InferenceConfigKey.LEFT_ARMQPOS.value: (-2.0, 2.0),  # 左臂7关节
            InferenceConfigKey.RIGHT_ARMQPOS.value: (-2.0, 2.0),  # 右臂7关节
            InferenceConfigKey.HEADQPOS.value: (-0.8, 0.8),  # 头部2关节
            InferenceConfigKey.ANKLEQPOS.value: (-0.5, 0.5),  # 脚踝
            InferenceConfigKey.KNEEQPOS.value: (0.0, 1.0),  # 膝盖
            InferenceConfigKey.BUTTOCKQPOS.value: (-0.5, 0.5),  # 臀部
            InferenceConfigKey.LEFT_EEFHAND.value: (-10.0, 110.0),  # 左手6个手指关节
            InferenceConfigKey.RIGHT_EEFHAND.value: (-10.0, 110.0),  # 右手6个手指关节
        }

        log_info("异步模拟客户端初始化完成")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def _parse_states_file(self, states_file_path):
        try:
            with open(states_file_path, "r") as f:
                content = f.read().strip()

            safe_globals = {
                "__builtins__": {},
                "array": np.array,
                "np": np,
                "float32": np.float32,
                "float64": np.float64,
            }

            try:
                states_dict = eval(content, safe_globals, {})

                for key, value in states_dict.items():
                    if isinstance(value, np.ndarray) and value.dtype != np.float32:
                        states_dict[key] = value.astype(np.float32)

                return states_dict
            except Exception as e:
                log_warning(f"解析states.txt失败: {e}")
                return None

        except Exception as e:
            log_error(f"读取states.txt文件失败: {e}")
            return None

    def load_observation_data_from_file(self, folder_path="debug_20251211_105644"):
        try:
            import glob
            from pathlib import Path
            import ast

            log_info(f"从本地数据文件夹加载观测数据: {folder_path}")

            batch_pattern = os.path.join(folder_path, "batch_*")
            batch_folders = sorted(glob.glob(batch_pattern))

            if not batch_folders:
                log_warning(f"未找到batch文件夹: {batch_pattern}")
                log_info("生成模拟数据")
                self._generate_mock_observations()
                return

            batch_folders = batch_folders[: min(len(batch_folders), self.max_steps)]

            for batch_idx, batch_folder in enumerate(batch_folders):
                try:
                    input_batch_path = os.path.join(batch_folder, "input_batch")

                    if not os.path.exists(input_batch_path):
                        log_warning(f"input_batch文件夹不存在: {input_batch_path}")
                        continue

                    obs = {
                        CommonKey.IMAGES.value: {},
                        CommonKey.STATES.value: {},
                        CommonKey.DISPS.value: [None],
                        CommonKey.TIMESTAMP.value: time.time() + batch_idx * 0.1,
                        CommonKey.TIMESTEP.value: batch_idx,
                        CommonKey.MUST_GO.value: (batch_idx == 0),
                        "start_infer": False,
                    }

                    image_files = sorted(
                        glob.glob(os.path.join(input_batch_path, "camera_*.png"))
                    )

                    camera_key_mapping = {
                        0: ImageKey.CAM_HIGH.value,  # camera_00.png
                        1: ImageKey.CAM_HIGH_R.value,  # camera_01.png
                        2: ImageKey.CAM_HAND_LEFT.value,  # camera_02.png
                        3: ImageKey.CAM_HAND_RIGHT.value,  # camera_03.png
                    }

                    for img_idx, img_file in enumerate(image_files):
                        if img_idx in camera_key_mapping:
                            # 读取图像
                            img = cv2.imread(img_file)
                            if img is not None:
                                camera_key = camera_key_mapping[img_idx]
                                obs[CommonKey.IMAGES.value][camera_key] = img
                            else:
                                log_warning(f"无法读取图像: {img_file}")

                    states_file = os.path.join(input_batch_path, "states.txt")
                    if os.path.exists(states_file):
                        states_dict = self._parse_states_file(states_file)

                        if states_dict:
                            expected_order = [
                                InferenceConfigKey.WAISTQPOS.value,
                                InferenceConfigKey.LEFT_ARMQPOS.value,
                                InferenceConfigKey.HEADQPOS.value,
                                InferenceConfigKey.RIGHT_ARMQPOS.value,
                                InferenceConfigKey.ANKLEQPOS.value,
                                InferenceConfigKey.KNEEQPOS.value,
                                InferenceConfigKey.BUTTOCKQPOS.value,
                                InferenceConfigKey.LEFT_EEFHAND.value,
                                InferenceConfigKey.RIGHT_EEFHAND.value,
                            ]

                            for key in expected_order:
                                if key in states_dict:
                                    obs[CommonKey.STATES.value][key] = states_dict[key]

                            joint_dims_and_defaults = {
                                InferenceConfigKey.WAISTQPOS.value: (1, 0.0),
                                InferenceConfigKey.HEADQPOS.value: (2, 0.0),
                                InferenceConfigKey.ANKLEQPOS.value: (1, 0.0),
                                InferenceConfigKey.KNEEQPOS.value: (1, 0.0),
                                InferenceConfigKey.BUTTOCKQPOS.value: (1, 0.0),
                            }

                            for joint_key, (
                                dim,
                                default_val,
                            ) in joint_dims_and_defaults.items():
                                if joint_key not in obs[CommonKey.STATES.value]:
                                    obs[CommonKey.STATES.value][joint_key] = np.full(
                                        (dim,), default_val, dtype=np.float32
                                    )
                        else:
                            log_warning(f"解析states.txt失败: {states_file}")
                    else:
                        log_warning(f"状态文件不存在: {states_file}")

                    if (
                        CommonKey.IMAGES.value in obs
                        and len(obs[CommonKey.IMAGES.value]) >= 2
                        and CommonKey.STATES.value in obs
                        and len(obs[CommonKey.STATES.value]) >= 4
                    ):

                        self.observation_data.append(obs)
                        log_info(f"成功加载batch #{batch_idx}: {batch_folder}")

                    else:
                        log_warning(f"batch #{batch_idx} 数据不完整，跳过")
                        log_warning(
                            f"  图像数量: {len(obs.get(CommonKey.IMAGES.value, {}))}"
                        )
                        log_warning(
                            f"  状态数量: {len(obs.get(CommonKey.STATES.value, {}))}"
                        )

                except Exception as batch_err:
                    log_error(f"加载batch #{batch_idx} 时出错: {batch_err}")
                    import traceback

                    traceback.print_exc()

            log_info(f"成功加载 {len(self.observation_data)} 份观测数据")

            if len(self.observation_data) == 0:
                log_warning("未加载到任何有效数据，生成模拟数据")
                self._generate_mock_observations()

        except Exception as e:
            log_error(f"加载观测数据失败: {e}")
            import traceback

            traceback.print_exc()
            log_info("生成模拟观测数据")
            self._generate_mock_observations()

    def _generate_mock_observations(self, save_path="test_observations.pkl"):
        log_info("生成模拟观测数据...")

        joint_dims = {
            InferenceConfigKey.WAISTQPOS.value: 1,
            InferenceConfigKey.LEFT_ARMQPOS.value: 7,
            InferenceConfigKey.HEADQPOS.value: 2,
            InferenceConfigKey.RIGHT_ARMQPOS.value: 7,
            InferenceConfigKey.ANKLEQPOS.value: 1,
            InferenceConfigKey.KNEEQPOS.value: 1,
            InferenceConfigKey.BUTTOCKQPOS.value: 1,
            InferenceConfigKey.LEFT_EEFHAND.value: 6,
            InferenceConfigKey.RIGHT_EEFHAND.value: 6,
        }

        for step in range(10):
            obs = {
                CommonKey.IMAGES.value: {},
                CommonKey.STATES.value: {},
                CommonKey.DISPS.value: [None],
                CommonKey.TIMESTAMP.value: time.time() + step * 0.1,
                CommonKey.TIMESTEP.value: step,
                CommonKey.MUST_GO.value: (step == 0),  # 第一步必须发送
                "start_infer": False,  # 默认不触发推理，由异步检查决定
            }

            head_w, head_h = self.head_target_size
            hand_w, hand_h = self.hand_target_size

            obs[CommonKey.IMAGES.value][ImageKey.CAM_HIGH.value] = np.random.randint(
                0, 256, (head_h, head_w, 3), dtype=np.uint8
            )
            obs[CommonKey.IMAGES.value][ImageKey.CAM_HIGH_R.value] = np.random.randint(
                0, 256, (head_h, head_w, 3), dtype=np.uint8
            )

            obs[CommonKey.IMAGES.value][
                ImageKey.CAM_HAND_LEFT.value
            ] = np.random.randint(0, 256, (hand_h, hand_w, 3), dtype=np.uint8)
            obs[CommonKey.IMAGES.value][
                ImageKey.CAM_HAND_RIGHT.value
            ] = np.random.randint(0, 256, (hand_h, hand_w, 3), dtype=np.uint8)

            states_config = {
                InferenceConfigKey.WAISTQPOS.value: (-30, 30),  # 腰部
                InferenceConfigKey.LEFT_ARMQPOS.value: (-180, 180),  # 左臂7关节
                InferenceConfigKey.HEADQPOS.value: (-60, 60),  # 颈部2关节
                InferenceConfigKey.RIGHT_ARMQPOS.value: (-180, 180),  # 右臂7关节
                InferenceConfigKey.ANKLEQPOS.value: (-30, 30),  # 脚踝
                InferenceConfigKey.KNEEQPOS.value: (0, 90),  # 膝盖
                InferenceConfigKey.BUTTOCKQPOS.value: (-15, 15),  # 臀部
                InferenceConfigKey.LEFT_EEFHAND.value: (0, 100),  # 左手6个手指关节
                InferenceConfigKey.RIGHT_EEFHAND.value: (0, 100),  # 右手6个手指关节
            }

            for key, (min_val, max_val) in states_config.items():
                dim = joint_dims[key]
                if key in [
                    InferenceConfigKey.LEFT_EEFHAND.value,
                    InferenceConfigKey.RIGHT_EEFHAND.value,
                ]:
                    # 手部关节用百分比 (0-100)
                    obs[CommonKey.STATES.value][key] = np.random.uniform(
                        min_val, max_val, (dim,)
                    ).astype(np.float32)
                else:
                    # 身体关节用角度
                    obs[CommonKey.STATES.value][key] = np.random.uniform(
                        min_val, max_val, (dim,)
                    ).astype(np.float32)

            self.observation_data.append(obs)

        try:
            with open(save_path, "wb") as f:
                pickle.dump(self.observation_data, f)
            log_info(f"模拟观测数据已保存到: {save_path}")
        except Exception as e:
            log_error(f"保存观测数据失败: {e}")

    def connect_to_server(self):
        log_info(f"连接到服务器: {self.server_host}:{self.server_port}")

        start_time = time.time()
        max_retries = 5
        retry_delay = 2.0

        for attempt in range(max_retries):
            if self.network_client.connect():
                self.connected = True
                log_info(f"成功连接到服务器 (尝试 {attempt+1}/{max_retries})")
                return True
            else:
                log_warning(f"连接失败，{retry_delay}秒后重试... (尝试 {attempt+1}/{max_retries})")
                time.sleep(retry_delay)

        log_error("无法连接到服务器")
        return False

    def _ready_to_send_observation(self):
        if not self.allow_infer_check:
            return

        queue_size = self.action_queue.qsize()

        # 如果动作队列为空，触发推理
        if queue_size == 0:
            self.start_infer = True
            self.allow_infer_check = False
            if self.current_step % 5 == 0:
                log_info("动作队列为空，触发推理")
                self.stats["inferences_triggered"] += 1
        elif (
            self.action_chunk_size > 0
            and queue_size / self.action_chunk_size <= self._chunk_size_threshold
        ):
            self.start_infer = True
            self.allow_infer_check = False
            log_info(f"队列长度 {queue_size} 低于阈值，触发推理")
            self.stats["inferences_triggered"] += 1

    def observation_collection_loop(self):
        self.start_barrier.wait()
        log_info("观测收集线程启动")

        collect_obs_interval = 1.0 / 2.0  # 2Hz
        next_obs_time = time.perf_counter()

        for step in range(self.max_steps):
            if not self.running:
                break

            loop_start = time.perf_counter()

            if step < len(self.observation_data):
                obs_data = self.observation_data[step]

                observation = TimedObservation(
                    timestamp=obs_data[CommonKey.TIMESTAMP.value],
                    observation=obs_data,
                    timestep=step,
                )

                observation.must_go = self.action_queue.empty()

                try:
                    self.observation_queue.put_nowait(observation)
                except:
                    try:
                        self.observation_queue.get_nowait()
                        self.observation_queue.put_nowait(observation)
                    except:
                        pass

                log_info(f"观测 #{step} 已收集")

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, collect_obs_interval - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

            next_obs_time += collect_obs_interval
            current_time = time.perf_counter()
            if current_time < next_obs_time:
                time.sleep(next_obs_time - current_time)
            else:
                next_obs_time = current_time

        log_info("观测收集线程完成")

    def observation_check_loop(self):
        self.start_barrier.wait()
        log_info("观测检查线程启动")

        check_interval = 1.0 / 5.0  # 5Hz
        next_check_time = time.perf_counter()

        while self.running and self.current_step < self.max_steps:
            loop_start = time.perf_counter()

            self._ready_to_send_observation()

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, check_interval - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

            next_check_time += check_interval
            current_time = time.perf_counter()
            if current_time < next_check_time:
                time.sleep(next_check_time - current_time)
            else:
                next_check_time = current_time

        log_info("观测检查线程完成")

    def observation_sender_loop(self):
        self.start_barrier.wait()
        log_info("观测发送线程启动")

        while self.running and self.current_step < self.max_steps:
            try:
                observation = self.observation_queue.get(timeout=0.1)

                success = self._send_observation_to_server(observation)

                if success:
                    self.stats["observations_sent"] += 1
                    self.current_step += 1

                    if observation.must_go:
                        log_info(f"观测 #{observation.get_timestep()} 已发送 (must_go=True)")

            except Empty:
                continue
            except Exception as e:
                log_error(f"观测发送错误: {e}")
                time.sleep(0.1)

        log_info("观测发送线程完成")

    def _send_observation_to_server(self, observation):
        try:
            obs_data = observation.get_observation()
            timestamp = observation.get_timestamp()
            timestep = observation.get_timestep()
            must_go = observation.must_go

            observation_dict = {
                CommonKey.TIMESTAMP.value: timestamp,
                CommonKey.TIMESTEP.value: timestep,
                CommonKey.MUST_GO.value: must_go,
                CommonKey.STATES.value: obs_data[CommonKey.STATES.value],
                "start_infer": self.start_infer,
            }

            if self.start_infer:
                log_info(f"发送推理请求，时间步: {timestep}")
                self.start_infer = False

            images = obs_data.get(CommonKey.IMAGES.value, {})
            for image_key, img_data in images.items():
                if img_data is not None:
                    compressed_img = compress_image(img_data)
                    observation_dict[image_key] = compressed_img

            start_time = time.perf_counter()
            response = self.network_client.send_request("observation", observation_dict)
            latency = (time.perf_counter() - start_time) * 1000

            self.stats["avg_observation_latency"] = (
                self.stats["avg_observation_latency"]
                * (self.stats["observations_sent"] - 1)
                + latency
            ) / max(1, self.stats["observations_sent"])

            if response and response.get("status") == "received":
                log_info(f"观测 #{timestep} 发送成功 (延迟: {latency:.2f}ms)")
                return True
            else:
                log_error(f"观测 #{timestep} 发送失败")
                return False

        except Exception as e:
            log_error(f"发送观测时出错: {e}")
            import traceback

            traceback.print_exc()
            return False

    def action_receiver_loop(self):
        self.start_barrier.wait()
        log_info("动作接收线程启动")

        get_actions_interval = 1.0 / 2.0
        next_recv_time = time.perf_counter()

        while self.running:
            try:
                loop_start = time.perf_counter()

                actions = self._get_actions_from_server()

                if actions is not None:
                    self.allow_infer_check = True
                    log_info("动作聚合完成，重新允许推理检查")

                    if "actions" in actions and "qpos" in actions["actions"]:
                        for key, action_list in actions["actions"]["qpos"].items():
                            if isinstance(action_list, list) and len(action_list) > 0:
                                self.action_chunk_size = max(
                                    self.action_chunk_size, len(action_list)
                                )

                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, get_actions_interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

                next_recv_time += get_actions_interval
                current_time = time.perf_counter()
                if current_time < next_recv_time:
                    time.sleep(next_recv_time - current_time)
                else:
                    next_recv_time = current_time

            except Exception as e:
                log_error(f"接收动作错误: {e}")
                time.sleep(0.1)

        log_info("动作接收线程完成")

    def _check_action_values(self, action_key, action_values):
        if action_key in self.joint_limits and isinstance(
            action_values, (list, np.ndarray)
        ):
            min_limit, max_limit = self.joint_limits[action_key]

            if isinstance(action_values, list):
                action_array = np.array(action_values)
            else:
                action_array = action_values

            assert not np.any(np.isnan(action_array)), f"动作 {action_key} 包含NaN值"
            assert not np.any(np.isinf(action_array)), f"动作 {action_key} 包含无穷大值"

            action_min = action_array.min()
            action_max = action_array.max()

            assert (
                action_min >= min_limit
            ), f"动作 {action_key} 最小值 {action_min:.2f} 低于限制 {min_limit:.2f}"
            assert (
                action_max <= max_limit
            ), f"动作 {action_key} 最大值 {action_max:.2f} 高于限制 {max_limit:.2f}"

            if len(action_array) > 0:
                below_min = np.sum(action_array < min_limit)
                above_max = np.sum(action_array > max_limit)
                total_values = len(action_array)

                if below_min > 0 or above_max > 0:
                    log_warning(
                        f"动作 {action_key} 中有 {below_min}/{total_values} 个值低于最小值, {above_max}/{total_values} 个值高于最大值"
                    )
                else:
                    log_info(
                        f"动作 {action_key} 值范围检查通过: {action_min:.2f} ~ {action_max:.2f} (范围: {min_limit:.2f} ~ {max_limit:.2f})"
                    )

            return True
        return False

    def _get_actions_from_server(self):
        try:
            start_time = time.perf_counter()
            response = self.network_client.send_request("get_actions")
            latency = (time.perf_counter() - start_time) * 1000

            assert response is not None, "获取动作响应为空"

            if response and response.get("status") == "success":
                actions = response.get("actions", {})
                timestamp = response.get(CommonKey.TIMESTAMP.value)
                timestep = response.get(CommonKey.TIMESTEP.value)

                log_info(f"成功获取动作 (时间步: {timestep}, 延迟: {latency:.2f}ms)")

                assert actions is not None, "动作为空"
                assert isinstance(actions, dict), f"动作类型错误: {type(actions)}"
                action_info = {
                    "timestep": timestep,
                    "timestamp": timestamp,
                    "latency": latency,
                    "actions": actions,
                }

                self.action_responses.append(action_info)
                self.stats["actions_received"] += 1

                self.stats["avg_action_latency"] = (
                    self.stats["avg_action_latency"]
                    * (self.stats["actions_received"] - 1)
                    + latency
                ) / max(1, self.stats["actions_received"])

                log_info("=" * 50)
                log_info(f"动作响应 #{len(self.action_responses)}")
                log_info(f"时间步: {timestep}")
                log_info(f"时间戳: {timestamp}")
                log_info(f"延迟: {latency:.2f}ms")
                if "qpos" in actions:
                    for key, joint_actions in actions["qpos"].items():
                        if isinstance(joint_actions, list):
                            log_info(f"  {key}: 长度={len(joint_actions)}")
                log_info("=" * 50)

                return action_info
            elif response and response.get("status") == "no_actions":
                log_info("服务器当前无可用动作")
                return None
            else:
                log_error("获取动作失败")
                return None

        except Exception as e:
            log_error(f"获取动作时出错: {e}")
            return None

    def run_async_test(self):
        if not self.connect_to_server():
            log_error("无法连接到服务器，测试终止")
            return False

        log_info("开始异步测试...")
        log_info(f"将发送 {len(self.observation_data)} 份观测数据")
        log_info("=" * 60)

        test_start_time = time.time()

        threads = []

        obs_collection_thread = threading.Thread(
            target=self.observation_collection_loop, name="ObservationCollection"
        )
        threads.append(obs_collection_thread)

        obs_check_thread = threading.Thread(
            target=self.observation_check_loop, name="ObservationCheck"
        )
        threads.append(obs_check_thread)

        obs_sender_thread = threading.Thread(
            target=self.observation_sender_loop, name="ObservationSender"
        )
        threads.append(obs_sender_thread)

        action_receiver_thread = threading.Thread(
            target=self.action_receiver_loop, name="ActionReceiver"
        )
        threads.append(action_receiver_thread)

        for thread in threads:
            thread.daemon = True
            thread.start()

        log_info("所有线程已启动，等待测试完成...")

        while self.current_step < self.max_steps and self.running:
            time.sleep(0.5)

        log_info("观测发送完成，等待动作处理...")
        time.sleep(3.0)
        self.shutdown_event.set()

        for thread in threads:
            thread.join(timeout=1.0)

        test_duration = time.time() - test_start_time

        assert self.stats["observations_sent"] > 0, "没有发送任何观测"
        assert (
            self.stats["observations_sent"] <= self.max_steps
        ), f"发送的观测数量超过最大值: {self.stats['observations_sent']}"

        if self.stats["inferences_triggered"] > 0:
            assert self.stats["actions_received"] > 0, "触发了推理但没有收到动作"
        for i, action_info in enumerate(self.action_responses):
            if "actions" in action_info and "qpos" in action_info["actions"]:
                qpos_actions = action_info["actions"]["qpos"]

                for action_key, action_values in qpos_actions.items():
                    if (
                        isinstance(action_values, (list, np.ndarray))
                        and len(action_values) > 0
                    ):
                        if action_key in self.joint_limits:
                            min_limit, max_limit = self.joint_limits[action_key]
                            if isinstance(action_values, list):
                                action_array = np.array(action_values)
                            else:
                                action_array = action_values

                            assert np.all(
                                action_array >= min_limit
                            ), f"最终检查: 动作响应 #{i+1} 的 {action_key} 有值低于最小值 {min_limit}"
                            assert np.all(
                                action_array <= max_limit
                            ), f"最终检查: 动作响应 #{i+1} 的 {action_key} 有值高于最大值 {max_limit}"

        self._print_test_summary(test_duration)

        return True

    def _print_test_summary(self, test_duration):
        log_info("=" * 60)
        log_info("异步测试完成")
        log_info("=" * 60)
        log_info(f"总耗时: {test_duration:.2f}秒")
        log_info(
            f"发送观测数: {self.stats['observations_sent']}/{len(self.observation_data)}"
        )
        log_info(f"获取动作数: {self.stats['actions_received']}")
        log_info(f"触发推理次数: {self.stats['inferences_triggered']}")
        log_info(f"平均观测延迟: {self.stats['avg_observation_latency']:.2f}ms")
        log_info(f"平均动作延迟: {self.stats['avg_action_latency']:.2f}ms")
        log_info(f"吞吐量: {self.stats['observations_sent']/test_duration:.2f} obs/秒")
        log_info("=" * 60)

        if self.action_responses:
            log_info("动作响应摘要:")
            for i, action_info in enumerate(self.action_responses):
                log_info(
                    f"  响应 #{i+1}: 时间步={action_info['timestep']}, 延迟={action_info['latency']:.2f}ms"
                )
                if "actions" in action_info and "qpos" in action_info["actions"]:
                    joint_count = len(action_info["actions"]["qpos"])
                    log_info(f"      关节数: {joint_count}")
        else:
            log_warning("未收到任何动作响应")

        log_info("=" * 60)

    def save_test_results(self, filename="async_test_results.json"):
        results = {
            "test_config": {
                "server_host": self.server_host,
                "server_port": self.server_port,
                "max_steps": self.max_steps,
            },
            "test_stats": self.stats,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_duration": None,
            "action_responses": self.action_responses,
        }

        if hasattr(self, "test_start_time"):
            results["test_duration"] = time.time() - self.test_start_time

        try:

            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            serializable_results = convert_numpy(results)

            with open(filename, "w") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            log_info(f"测试结果已保存到: {filename}")
            return True
        except Exception as e:
            log_error(f"保存测试结果失败: {e}")
            return False

    def stop(self):
        log_info("停止异步模拟客户端...")
        self.shutdown_event.set()
        self.network_client.close()


def run_single_test(args, model_config):
    log_info("=" * 60)
    log_info("开始单个异步测试")
    log_info("=" * 60)

    client = AsyncMockRobotClient(
        args, model_config, args.server_host, args.server_port
    )

    if args.generate_data or not os.path.exists(args.data_file):
        client._generate_mock_observations(args.data_file)
    else:
        client.load_observation_data_from_file(args.data_file)

    try:
        success = client.run_async_test()

        if success:
            log_info("异步测试成功完成")
            client.save_test_results()
        else:
            log_error("异步测试失败")

    except KeyboardInterrupt:
        log_info("测试被用户中断")
    except Exception as e:
        log_error(f"测试运行时出错: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.stop()

    return client


def run_comprehensive_test(args, model_config):
    log_info("=" * 60)
    log_info("开始综合稳定性测试")
    log_info("=" * 60)

    all_results = []

    for test_run in range(args.num_runs):
        log_info(f"\n运行测试 #{test_run + 1}/{args.num_runs}")
        log_info("-" * 40)

        client = AsyncMockRobotClient(
            args, model_config, args.server_host, args.server_port
        )

        client._generate_mock_observations(f"test_data_run_{test_run}.pkl")

        try:
            start_time = time.time()
            success = client.run_async_test()
            duration = time.time() - start_time

            if success:
                run_result = {
                    "run_id": test_run + 1,
                    "duration": duration,
                    "stats": client.stats,
                    "success": True,
                }
                all_results.append(run_result)

                log_info(f"测试 #{test_run + 1} 完成: 成功")
            else:
                run_result = {
                    "run_id": test_run + 1,
                    "duration": duration,
                    "success": False,
                }
                all_results.append(run_result)

                log_error(f"测试 #{test_run + 1} 完成: 失败")

        except Exception as e:
            log_error(f"测试 #{test_run + 1} 异常: {e}")
            all_results.append(
                {"run_id": test_run + 1, "error": str(e), "success": False}
            )
        finally:
            client.stop()

        if test_run < args.num_runs - 1:
            log_info(f"等待 {args.run_interval} 秒后开始下一次测试...")
            time.sleep(args.run_interval)

    _analyze_comprehensive_results(all_results, args)

    return all_results


def _analyze_comprehensive_results(all_results, args):
    log_info("\n" + "=" * 60)
    log_info("综合稳定性测试结果分析")
    log_info("=" * 60)

    successful_runs = [r for r in all_results if r.get("success", False)]
    failed_runs = [r for r in all_results if not r.get("success", False)]

    log_info(f"总运行次数: {len(all_results)}")
    log_info(f"成功次数: {len(successful_runs)}")
    log_info(f"失败次数: {len(failed_runs)}")
    log_info(f"成功率: {len(successful_runs)/len(all_results)*100:.1f}%")

    if successful_runs:
        avg_stats = {}
        for key in ["observations_sent", "actions_received", "inferences_triggered"]:
            values = [run["stats"].get(key, 0) for run in successful_runs]
            avg_stats[key] = sum(values) / len(values)

        log_info("\n平均统计信息:")
        for key, value in avg_stats.items():
            log_info(f"  {key}: {value:.2f}")

        obs_latencies = [
            run["stats"].get("avg_observation_latency", 0) for run in successful_runs
        ]
        action_latencies = [
            run["stats"].get("avg_action_latency", 0) for run in successful_runs
        ]

        if obs_latencies:
            log_info(f"  平均观测延迟: {sum(obs_latencies)/len(obs_latencies):.2f}ms")
        if action_latencies:
            log_info(f"  平均动作延迟: {sum(action_latencies)/len(action_latencies):.2f}ms")

    if failed_runs:
        log_info("\n失败运行详情:")
        for run in failed_runs:
            log_info(f"  运行 #{run['run_id']}: {run.get('error', '未知错误')}")

    log_info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="异步模拟客户端测试")
    parser.add_argument(
        "--server_host", type=str, default="127.0.0.1", help="策略服务器主机地址"
    )
    parser.add_argument("--server_port", type=int, default=8889, help="策略服务器端口")
    parser.add_argument("--model_config", type=str, required=False, help="模型配置文件路径")
    parser.add_argument(
        "--data_file",
        type=str,
        default="test_observations.pkl",
        help="观测数据文件路径",
    )
    parser.add_argument("--generate_data", action="store_true", help="是否生成模拟数据")
    parser.add_argument(
        "--test_mode",
        type=str,
        default="single",
        choices=["single", "comprehensive"],
        help="测试模式: single(单次), comprehensive(综合)",
    )
    parser.add_argument("--num_runs", type=int, default=3, help="综合测试运行次数")
    parser.add_argument("--run_interval", type=float, default=2.0, help="综合测试运行间隔(秒)")

    args = parser.parse_args()

    model_config = {}
    if args.model_config:
        try:
            with open(args.model_config, "r") as f:
                model_config = json.load(f)
            log_info(f"加载模型配置: {args.model_config}")
        except Exception as e:
            log_error(f"加载模型配置失败: {e}")
            model_config = {}

    if args.test_mode == "single":
        client = run_single_test(args, model_config)
        return 0 if client.stats.get("actions_received", 0) > 0 else 1
    else:
        results = run_comprehensive_test(args, model_config)
        successful_runs = [r for r in results if r.get("success", False)]
        return 0 if len(successful_runs) > 0 else 1


if __name__ == "__main__":

    log_info("=" * 60)
    log_info("异步模拟客户端测试")
    log_info("模拟真实客户端的异步推理触发机制")
    log_info("注意：请确保策略服务器已在运行")
    log_info("=" * 60)

    exit_code = main()

    log_info("=" * 60)
    log_info("测试程序结束")
    log_info("=" * 60)

    sys.exit(exit_code)
