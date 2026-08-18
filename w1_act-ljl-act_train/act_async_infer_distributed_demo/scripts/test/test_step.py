import torch
import numpy as np
import time
import os
import cv2
import argparse
from pathlib import Path
import json
import glob
from collections import deque
from copy import deepcopy

from act_async_infer_distributed_demo.scripts.utils_distributed import (
    TimedObservation,
    log_info,
    log_error,
    log_warning,
    normalize_array,
    Debug_Datasaver,
    set_seed
)

from dexechain.data.enum import (
    Modality,
    SUPPORTED_PROPRIO_TYPES,
    JointType,
)

from act_async_infer_distributed_demo.scripts.w1_mapping import (
    CommonKey,
    InferenceConfigKey,
    DeployedDexforceVLAKey,
    w1qpos_names_map,
    EnumIndex,
)

from scipy.ndimage import gaussian_filter1d



_DEFAULT_MODEL_CONFIG = {
    "action_horizon": 64,
    "hand_sim_max_limit": [87.27, 157.08, 130.9, 130.9, 130.9, 130.9],
    "gripper_sim_max_limit": [],
    "data_type": "sim",
    "is_gripper_bool": True,
    "model_end_effector_limit": [0.0, 1.0],
    "end_effector_limit": [0, 100],
    "inverse_gripper": False,
    "use_post_opt": True,
}



def load_model_config(config_path: str) -> dict:
    """从 JSON 加载模型配置，未指定的字段使用默认值"""
    with open(config_path, "r") as f:
        user_config = json.load(f)
    merged = {**_DEFAULT_MODEL_CONFIG, **user_config}
    merged[DeployedDexforceVLAKey.MODEL_NAME.value] = merged.pop("model_name", "act")
    merged[DeployedDexforceVLAKey.PRETRAINED.value] = merged.pop("pretrained", "")
    return merged



class PredictActionChunkTester:
    def __init__(self, model_config, args, data_folder_path=None):
        self.model_config = model_config
        self.args = args
        self.data_folder_path = data_folder_path

        self.action_horizon = model_config.get("action_horizon", 64)

        self.init_flag = True
        self.first_prediction = True
        self.step_time = deque(maxlen=50)
        self.current_session_id = 1

        self.left_hand_qpos_names = w1qpos_names_map.get(
            InferenceConfigKey.LEFT_EEFHAND.value, []
        )
        self.right_hand_qpos_names = w1qpos_names_map.get(
            InferenceConfigKey.RIGHT_EEFHAND.value, []
        )
        self.hand_pos_len = len(self.left_hand_qpos_names) + len(
            self.right_hand_qpos_names
        )

        self.hand_sim_max_limit = (
            np.array(
                model_config.get(
                    "hand_sim_max_limit", [87.27, 157.08, 130.9, 130.9, 130.9, 130.9]
                )
            )
            / 100.0
        )
        self.gripper_types_to_process = model_config.get(
            "gripper_types_to_process",
            EnumIndex.gripper_types_to_process.value,
        )
        self.end_effector_limit = model_config.get("end_effector_limit", [0, 100])
        self.model_end_effector_limit = model_config.get("model_end_effector_limit", [0.0, 1.0])
        self.inverse_gripper = model_config.get("inverse_gripper", False)
        self.data_type = model_config.get("data_type", "sim")

        from act_async_infer_distributed_demo.scripts.w1_mapping import w1qpos_group_map
        self.state_keys = []
        for key in w1qpos_group_map.keys():
            self.state_keys.append(key)

        set_seed()
        self._initialize_real_model()

        self.observation_data = []
        self.debug_saver = None
        if data_folder_path:
            self._load_observation_data_from_folder(data_folder_path)


    def _initialize_real_model(self):
        log_info("正在初始化真实模型...")

        try:
            from dexechain.toolkits.vla import create_model
            from dexechain.toolkits.vla import DeployedDexforceVLA

            self.model: DeployedDexforceVLA = create_model(
                model_name=self.model_config[DeployedDexforceVLAKey.MODEL_NAME.value],
                pretrained=self.model_config[DeployedDexforceVLAKey.PRETRAINED.value],
                precompute_lang_embeddings=self.model_config.get(
                    DeployedDexforceVLAKey.PRECOMPUTE_LANG_EMBEDDINGS.value, None
                ),
                action_horizon=self.action_horizon,
            )

            self.model.is_gripper_bool = self.model_config.get("is_gripper_bool", True)

            log_info(f"✓ 模型初始化成功: {type(self.model).__name__}")
            log_info(f"✓ 使用的相机: {self.model.policy.camera_used}")
            log_info(f"✓ 状态元数据: {self.model.policy.state_meta}")
            log_info(f"✓ 动作序列长度: {self.model.action_horizon}")
            log_info(f"✓ 相机数量: {len(self.model.policy.camera_used)}")

            self.model.reset()

        except Exception as e:
            log_error(f"✗ 模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise


    def _load_observation_data_from_folder(self, folder_path):
        """
        模型已加载完成，self.model.policy.camera_used 可用。
        加载图像时按 camera_used 的顺序读取。
        """
        try:
            log_info(f"从本地数据文件夹加载观测数据: {folder_path}")

            batch_pattern = os.path.join(folder_path, "batch_*")
            batch_folders = sorted(glob.glob(batch_pattern))

            if not batch_folders:
                log_warning(f"未找到batch文件夹: {batch_pattern}")
                return

            max_steps = min(10, len(batch_folders))
            batch_folders = batch_folders[:max_steps]

            for batch_idx, batch_folder in enumerate(batch_folders):
                try:
                    input_batch_path = os.path.join(batch_folder, "input_batch")
                    if not os.path.exists(input_batch_path):
                        log_warning(f"input_batch文件夹不存在: {input_batch_path}")
                        continue

                    image_files = sorted(
                        glob.glob(os.path.join(input_batch_path, "camera_*.png"))
                    )
                    num_available_images = len(image_files)
                    num_expected_cameras = len(self.model.policy.camera_used)

                    if num_available_images != num_expected_cameras:
                        log_error(
                            f"batch #{batch_idx}: "
                            f"图像数量 ({num_available_images}) ≠ 模型期望相机数 ({num_expected_cameras})"
                        )
                        log_error(f"   模型 camera_used: {self.model.policy.camera_used}")
                        log_error(f"   存在的图像文件: {[os.path.basename(f) for f in image_files]}")
                        log_error("   终止测试！")
                        raise SystemExit(
                            f"Camera count mismatch: "
                            f"{num_available_images} images vs {num_expected_cameras} expected"
                        )

                    obs_images = {}
                    for img_idx, camera_key in enumerate(self.model.policy.camera_used):
                        img = cv2.imread(image_files[img_idx])
                        if img is not None:
                            obs_images[camera_key] = img
                        else:
                            log_warning(
                                f"无法读取图像 [{img_idx}] {image_files[img_idx]} "
                                f"→ 相机 {camera_key}"
                            )

                    obs = {
                        CommonKey.IMAGES.value: obs_images,
                        CommonKey.STATES.value: {},
                        CommonKey.DISPS.value: [None],
                        CommonKey.TIMESTAMP.value: time.time() + batch_idx * 0.1,
                        CommonKey.TIMESTEP.value: batch_idx,
                        CommonKey.MUST_GO.value: (batch_idx == 0),
                        CommonKey.INSTRUCTION.value: "",
                    }

                    states_file = os.path.join(input_batch_path, "states.txt")
                    if os.path.exists(states_file):
                        states_dict = self._parse_states_file(states_file)
                        if states_dict:
                            for key in self.state_keys:
                                if key in states_dict:
                                    obs[CommonKey.STATES.value][key] = states_dict[key]
                        else:
                            log_warning(f"解析states.txt失败: {states_file}")
                    else:
                        log_warning(f"状态文件不存在: {states_file}")
                        
                    prompt_file = os.path.join(input_batch_path, "prompt.txt")
                    if os.path.exists(prompt_file):
                        prompt_dict = self._parse_states_file(prompt_file)
                        if prompt_dict:
                            obs[CommonKey.INSTRUCTION.value] = prompt_dict["prompt"] if "prompt" in prompt_dict else ['']
                        else:
                            log_warning(f"prompt.txt失败: {prompt_file}")
                    else:
                        log_warning(f"语言提示词文件不存在: {prompt_file}")

                    if (
                        CommonKey.IMAGES.value in obs
                        and CommonKey.STATES.value in obs
                    ):
                        self.observation_data.append(obs)
                        log_info(
                            f"✓ 成功加载batch #{batch_idx} "
                            f"(图像: {list(obs[CommonKey.IMAGES.value].keys())}, "
                            f"状态: {list(obs[CommonKey.STATES.value].keys())})"
                        )
                    else:
                        log_warning(f"batch #{batch_idx} 数据不完整，跳过")

                except SystemExit:
                    raise
                except Exception as batch_err:
                    log_error(f"加载batch #{batch_idx} 时出错: {batch_err}")

            log_info(f"成功加载 {len(self.observation_data)} 份观测数据")

            if len(self.observation_data) == 0:
                log_error("未加载到任何有效数据")

        except SystemExit:
            raise
        except Exception as e:
            log_error(f"加载观测数据失败: {e}")
            import traceback
            traceback.print_exc()

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
            states_dict = eval(content, safe_globals, {})
            for key, value in states_dict.items():
                if isinstance(value, np.ndarray) and value.dtype != np.float32:
                    states_dict[key] = value.astype(np.float32)
            return states_dict
        except Exception as e:
            log_warning(f"解析states.txt失败: {e}")
            return None



    def _map_obs_to_model(self, observation_t):
        """使用 normalize_array 进行归一化，支持 hand 和 gripper。"""
        observation = observation_t.get_observation()

        multiplier = (
            1.0
            if self.model_config.get("data_type", "sim") == "real"
            else self.hand_sim_max_limit
        )

        for type_idx in self.gripper_types_to_process:
            proprio_type = SUPPORTED_PROPRIO_TYPES[type_idx]
            if proprio_type in observation[Modality.STATES.value]:
                # 手部（SUPPORTED_PROPRIO_TYPES[4], [5]）
                if (
                    proprio_type == SUPPORTED_PROPRIO_TYPES[4]
                    or proprio_type == SUPPORTED_PROPRIO_TYPES[5]
                ):
                    observation[Modality.STATES.value][proprio_type] = (
                        normalize_array(
                            observation[Modality.STATES.value][proprio_type],
                            src_min=self.end_effector_limit[0],
                            src_max=self.end_effector_limit[1],
                            dst_min=self.model_end_effector_limit[0],
                            dst_max=self.model_end_effector_limit[1],
                            reverse=self.inverse_gripper,
                        )
                        * multiplier
                    ).astype(np.float32)

                # 二指夹爪（SUPPORTED_PROPRIO_TYPES[6], [7]）
                elif (
                    proprio_type == SUPPORTED_PROPRIO_TYPES[6]
                    or proprio_type == SUPPORTED_PROPRIO_TYPES[7]
                ):
                    observation[Modality.STATES.value][proprio_type] = normalize_array(
                        observation[Modality.STATES.value][proprio_type],
                        src_min=self.end_effector_limit[0],
                        src_max=self.end_effector_limit[1],
                        dst_min=self.model_end_effector_limit[0],
                        dst_max=self.model_end_effector_limit[1],
                        reverse=self.inverse_gripper,
                    )

        for key in list(observation[Modality.STATES.value].keys()):
            if key not in self.model.policy.state_meta:
                del observation[Modality.STATES.value][key]
        instruction = observation.get(CommonKey.INSTRUCTION.value, "")

        obs = {
            Modality.IMAGES.value: [
                observation[Modality.IMAGES.value][cam_name]
                for cam_name in self.model.policy.camera_used
            ],
            Modality.STATES.value: observation[Modality.STATES.value],
            CommonKey.DISPS.value: observation[CommonKey.DISPS.value],
        }
        if instruction:
            obs[CommonKey.INSTRUCTION.value] = instruction
        log_info(f"✓ 成功构建观测数据")
        log_info(f"  - 图像数量: {len(obs[Modality.IMAGES.value])}")
        log_info(f"  - 状态键: {list(obs[Modality.STATES.value].keys())}")

        return obs

    def _predict_action_chunk(self, observation_t, debug_saver=None):
        """
        执行推理，返回 action_chunk_np。
        如果 debug_saver 不为 None，则保存调试数据。
        """
        log_info(
            f"Session #{self.current_session_id} - Predicting action chunk "
            f"for observation #{observation_t.get_timestep()}"
        )

        batch = self._map_obs_to_model(observation_t)


        batch_map = self.model.from_real_obs(deepcopy(batch), init=self.init_flag)
        self.init_flag = False

        if batch is None:
            raise RuntimeError("Failed to build observation")


        try:
            with torch.no_grad():
                start_predict = time.perf_counter()

                if CommonKey.INSTRUCTION.value in batch:
                    batch_map[Modality.LANG.value] = batch[CommonKey.INSTRUCTION.value]
                else:
                    batch_map[Modality.LANG.value] = ['']


                action_chunk = self.model.step(**batch_map)


                if debug_saver is not None:
                    if self.args.save_input:
                        try:
                            debug_saver.save_batch(
                                images=batch[Modality.IMAGES.value],
                                states_value=batch[Modality.STATES.value],
                                prompt_value=batch[CommonKey.INSTRUCTION.value] if CommonKey.INSTRUCTION.value in batch else "",
                            )
                        except Exception as e:
                            log_error(f"Error saving input data for debugging: {e}")
                    if self.args.save_vis_images:
                        debug_saver._save_vis_images(self.model.vis_images)
                    if self.args.save_joint_change:
                        # 传入历史 pickle 路径做对比
                        debug_saver.plot_joint_changes(
                            action_chunk[JointType.QPOS.value],
                            prev_pickle_path=self.args.prev_pickle,
                        )
                        # 保存本次 action_chunk 为 pickle，以便后续对比分析使用
                        debug_saver._save_action_chunk_pkl(action_chunk[JointType.QPOS.value])


                if self.args.use_smooth:
                    for key in action_chunk[JointType.QPOS.value].keys():
                        action_chunk[JointType.QPOS.value][key] = gaussian_filter1d(
                            action_chunk[JointType.QPOS.value][key],
                            sigma=2,
                            axis=0,
                            mode="nearest",
                        )

                # 反归一化
                multiplier = (
                    1.0
                    if self.model_config.get("data_type", "sim") == "real"
                    else self.hand_sim_max_limit
                )
                for type_idx in self.gripper_types_to_process:
                    proprio_type = SUPPORTED_PROPRIO_TYPES[type_idx]
                    if proprio_type in action_chunk[JointType.QPOS.value].keys():
                        for action in action_chunk[JointType.QPOS.value][proprio_type]:
                            if (
                                proprio_type == SUPPORTED_PROPRIO_TYPES[4]
                                or proprio_type == SUPPORTED_PROPRIO_TYPES[5]
                            ):
                                normalized = normalize_array(
                                    action,
                                    src_min=self.model_end_effector_limit[0],
                                    src_max=self.model_end_effector_limit[1],
                                    dst_min=self.end_effector_limit[0],
                                    dst_max=self.end_effector_limit[1],
                                    reverse=self.inverse_gripper,
                                ) * (1 / multiplier)
                                action[:] = normalized[:]
                            elif (
                                proprio_type == SUPPORTED_PROPRIO_TYPES[6]
                                or proprio_type == SUPPORTED_PROPRIO_TYPES[7]
                            ):
                                normalized = normalize_array(
                                    action,
                                    src_min=self.model_end_effector_limit[0],
                                    src_max=self.model_end_effector_limit[1],
                                    dst_min=self.end_effector_limit[0],
                                    dst_max=self.end_effector_limit[1],
                                    reverse=self.inverse_gripper,
                                )
                                action[:] = normalized[:]






                finish_predict = (time.perf_counter() - start_predict) * 1000
                self.step_time.append(finish_predict)
                avg_time = np.mean(self.step_time)
                log_warning(f"Step 时间: {finish_predict:.2f}ms")
                log_warning(f"平均推理时间: {avg_time:.2f}ms")

            action_chunk_np = {}
            for key, value in action_chunk.items():
                if isinstance(value, dict):
                    action_chunk_np[key] = {
                        k: v.numpy() if torch.is_tensor(v) else v
                        for k, v in value.items()
                    }
                elif torch.is_tensor(value):
                    action_chunk_np[key] = value.numpy()
                else:
                    action_chunk_np[key] = value

        except Exception as e:
            log_error(f"Error in policy.select_action: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Inference failed: {e}")

        return action_chunk_np


    def test_predict_action_chunk_performance(self, num_tests=None):
        if num_tests is None:
            num_tests = len(self.observation_data)
        else:
            num_tests = min(num_tests, len(self.observation_data))

        log_info(f"\n{'='*60}")
        log_info(f"开始 _predict_action_chunk 性能测试")
        log_info(f"测试次数: {num_tests}")
        log_info(f"{'='*60}")

        if (
            self.args.save_input
            or self.args.save_vis_images
            or self.args.save_joint_change
        ):
            self.debug_saver = Debug_Datasaver(
                camera_used=self.model.policy.camera_used
            )
            log_info("✓ Debug_Datasaver 已初始化")
        else:
            self.debug_saver = None

        execution_times = []
        successful_tests = 0

        for test_idx in range(num_tests):
            log_info(f"\n--- 测试 {test_idx + 1}/{num_tests} ---")

            try:
                self.first_prediction = True
                obs_data = self.observation_data[test_idx]
                observation = TimedObservation(
                    timestamp=obs_data[CommonKey.TIMESTAMP.value],
                    timestep=obs_data[CommonKey.TIMESTEP.value],
                    observation=obs_data,
                    must_go=obs_data[CommonKey.MUST_GO.value],
                )

                start_time = time.perf_counter()
                action_chunk = self._predict_action_chunk(
                    observation, debug_saver=self.debug_saver
                )
                end_time = time.perf_counter()

                execution_time = (end_time - start_time) * 1000
                execution_times.append(execution_time)
                successful_tests += 1

                log_info(f"✓ 测试 {test_idx + 1} 完成")
                log_info(f"  执行时间: {execution_time:.2f} ms")
                log_info(f"  动作序列键: {list(action_chunk.keys())}")

                if JointType.QPOS.value in action_chunk:
                    for key, values in action_chunk[JointType.QPOS.value].items():
                        if isinstance(values, np.ndarray):
                            log_info(f"    {key}: 形状={values.shape}")
                        elif isinstance(values, list):
                            log_info(f"    {key}: 长度={len(values)}")

            except Exception as e:
                log_info(f"✗ 测试 {test_idx + 1} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if execution_times:
            avg_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)

            log_info(f"\n{'='*60}")
            log_info(f"性能测试结果摘要")
            log_info(f"{'='*60}")
            log_info(f"成功测试次数: {successful_tests}/{num_tests}")
            log_info(f"平均执行时间: {avg_time:.2f} ± {std_time:.2f} ms")
            log_info(f"最小执行时间: {min_time:.2f} ms")
            log_info(f"最大执行时间: {max_time:.2f} ms")
            log_info(f"时间标准差: {std_time:.2f} ms")

            results_file = "predict_action_chunk_performance_results.txt"
            with open(results_file, "w") as f:
                f.write("_predict_action_chunk 性能测试结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"成功测试次数: {successful_tests}/{num_tests}\n")
                f.write(f"平均执行时间: {avg_time:.2f} ± {std_time:.2f} ms\n")
                f.write(f"最小执行时间: {min_time:.2f} ms\n")
                f.write(f"最大执行时间: {max_time:.2f} ms\n")
                f.write(f"时间标准差: {std_time:.2f} ms\n")
                f.write("\n详细执行时间:\n")
                for i, t in enumerate(execution_times):
                    f.write(f"  测试 {i+1}: {t:.2f} ms\n")

            log_info(f"✓ 详细结果已保存到: {results_file}")

            return {
                "successful_tests": successful_tests,
                "avg_time": avg_time,
                "std_time": std_time,
                "min_time": min_time,
                "max_time": max_time,
                "all_times": execution_times,
            }
        else:
            log_info("✗ 没有成功的测试")
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="模型配置 JSON 文件路径",
    )
    parser.add_argument(
        "--data_folder_path",
        type=str,
        default=None,
        help="观测数据文件夹路径",
    )
    parser.add_argument("--num_tests", type=int, default=1, help="测试次数")
    parser.add_argument("--use_smooth", action="store_true", help="是否使用平滑")
    parser.add_argument("--save_input", action="store_true", help="保存输入数据")
    parser.add_argument("--save_vis_images", action="store_true", help="保存可视化图像")
    parser.add_argument("--save_joint_change", action="store_true", help="保存关节变化")
    parser.add_argument("--prev_pickle", type=str, default=None,help="历史 pickle 路径，用于对比画图")


    args = parser.parse_args()

    if args.config:
        model_config = load_model_config(args.config)
        log_info(f"✓ 从 {args.config} 加载配置")
    else:
        model_config = {**_DEFAULT_MODEL_CONFIG}
        model_config[DeployedDexforceVLAKey.MODEL_NAME.value] = model_config.pop(
            "model_name", "act"
        )
        model_config[DeployedDexforceVLAKey.PRETRAINED.value] = model_config.pop(
            "pretrained", ""
        )
        log_info("✓ 使用默认配置（未指定 --config）")

    log_info(f"  模型: {model_config[DeployedDexforceVLAKey.MODEL_NAME.value]}")
    log_info(f"  权重: {model_config[DeployedDexforceVLAKey.PRETRAINED.value]}")
    log_info(f"  action_horizon: {model_config['action_horizon']}")
    log_info(f"  data_type: {model_config['data_type']}")

    log_info("初始化测试器...")
    tester = PredictActionChunkTester(
        model_config=model_config,
        args=args,
        data_folder_path=args.data_folder_path,
    )

    results = tester.test_predict_action_chunk_performance(num_tests=args.num_tests)

    if results:
        log_info(f"\n🎉 测试完成!")
        log_info(f"平均推理时间: {results['avg_time']:.2f} ms")
    else:
        log_info(f"\n❌ 测试失败!")


if __name__ == "__main__":
    main()
