import argparse
import threading
import time
import numpy as np
from queue import Empty, Queue
from collections import OrderedDict
from typing import Dict
import torch
from collections import deque
from copy import deepcopy
import traceback
from scipy.ndimage import gaussian_filter1d
from act_async_infer_distributed_demo.scripts.w1_mapping import (
    DeployedDexforceVLAKey,
    InferenceConfigKey,
    CommonKey,
    w1qpos_names_map,
    EnumIndex
)
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    TimedObservation,
)
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_info,
    log_error,
    log_warning,
    log_debug,
    setup_logger,
    Debug_Datasaver,
    RealTimeFrequencyMonitor,
    normalize_array,
    set_seed
)
from act_async_infer_distributed_demo.scripts.inference_config import (
    ServerConfig, 
    RequestType,
    ResponseKey
)
from act_async_infer_distributed_demo.scripts.network_utils import (
    NetworkServer,
)
try:
    import dexechain
except Exception as e:
    log_error(
        "Fail to import dexechain. Check if dexechain is installed. Reference command: cd embodichain && pip install -e .[deploy]"
    )

from dexechain.data.enum import (
    Modality,
    SUPPORTED_PROPRIO_TYPES,
    JointType,
)

np.set_printoptions(precision=4, suppress=True, linewidth=100)


class PolicyServerPureInference:
    _logger_initialized = False
    def __init__(self, cfg: ServerConfig, server_host: str, server_port: int):
        self.cfg = cfg
        self.server_host = server_host
        self.server_port = server_port
        self.past_policy_path = self.cfg.pretrained
        self.init_flag = True
        self.origin_joint_names = None
        self.tracer = None
        self.left_hand_qpos_names = w1qpos_names_map[InferenceConfigKey.LEFT_EEFHAND.value]
        self.right_hand_qpos_names = w1qpos_names_map[InferenceConfigKey.RIGHT_EEFHAND.value]
        # 网络服务器
        self.network_server = NetworkServer(server_host, server_port)
        self.first_connect = True
        self.network_server.set_disconnect_callback(self._on_client_disconnected)
        # 推理参数
        self.obs_queue_timeout = 0.001
        self.inference_latency = 0.0
        self.inference_output_body_dof = None
        self.count = 0
        self.step_time = deque(maxlen=50)
        # 推理状态
        self.shutdown_event = threading.Event()
        self.ready_event = threading.Event()
        self.actions_ready_event = threading.Event()
        self.start_infer_event = threading.Event()
        self.busy_event = threading.Event()
        # 观测队列系统
        self.observation_queue = Queue(maxsize=1)
        self.past_obs_queue = Queue(maxsize=50)
        self.last_processed_obs = None
        self.observation_queue_lock = threading.Lock()
        self.latest_queued_timestep = -1
        self.Debug_Datasaver = None
        # 当前动作数据
        self._predicted_timesteps = set()
        self.current_actions = None
        self.current_timestamp = 0.0
        self.current_timestep = 0
        self.current_actions_lock = threading.Lock()
        # 动作数据队列系统
        self.actions_queue = Queue(maxsize=1)
        self.actions_metadata_queue = Queue(maxsize=1)
        self.cfg.hand_sim_max_limit = (
            np.array(cfg.hand_sim_max_limit) / 100.0
        )
        self.gripper_types_to_process = EnumIndex.gripper_types_to_process.value
        self.end_effector_limit = None
        # 模型状态
        self.model = None
        self._model_none_warned = False
        self._debug_pool = None
        self.first_prediction = True
        self.status_message = ''
        # 客户端连接状态
        self.client_connected = False
        self.client_connection_lock = threading.Lock()
        self.last_client_activity = time.time()
        # 查看函数调用频率
        self.monitor = RealTimeFrequencyMonitor(window_size=60)
        # 数据记录和可视化
        self.actions = OrderedDict()
        self.actionchunks = []
        # 会话管理
        self.current_session_id = 0
        # 配置更新标志
        self.config_update_pending = False
        self.on_client_disconnected_callback = None
        self.new_config = None
        self.setup_config_flag = False
        self.state = "idle"
        self.error_message = None
        # 注册请求处理器
        self.network_server.register_handler(RequestType.OBSERVATION, self._handle_observation)
        self.network_server.register_handler(RequestType.GET_ACTIONS, self._handle_get_actions)
        self.network_server.register_handler(RequestType.SETUP_CONFIG, self._handle_control_command)
        self.network_server.register_handler(RequestType.STOP, self._handle_control_command)
        self.network_server.register_handler(RequestType.STATUS, self._handle_control_command)
        self.network_server.register_handler(RequestType.SHUTDOWN, self._handle_control_command)
        set_seed()
        self.start()

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def _initialize_model(self):
        log_info("Starting DexforceVLA model initialization...")
        from dexechain.toolkits.vla import create_model
        from dexechain.toolkits.vla import DeployedDexforceVLA

        
        # 记录使用的配置
        log_debug(f"Loading model: {self.cfg.model_name}")
        log_debug(f"Pretrained path: {self.cfg.pretrained}")
        log_debug(f"Action horizon: {self.cfg.action_horizon}")
        
        self.model = None
        self.model: DeployedDexforceVLA = create_model(
            model_name=self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            precompute_lang_embeddings=None,
            action_horizon=self.cfg.action_horizon,
        )
        
        # 模型加载成功，重置警告标志
        self._model_none_warned = False  


        if (
            self.cfg.save_input
            or self.cfg.save_vis_images
            or self.cfg.save_joint_change
        ):
            self.Debug_Datasaver = Debug_Datasaver(
                camera_used=self.model.policy.camera_used
            )
        self.model.is_gripper_bool = self.cfg.is_gripper_bool

        log_debug(f"Save input: {self.cfg.save_input}.")
        log_debug(f"Save vis_image: {self.cfg.save_vis_images}.")
        log_debug(f"Save joint_change: {self.cfg.save_joint_change}.")
        log_debug(f"is_gripper_bool: {self.model.is_gripper_bool}.")
        log_debug(f"model qpos type:{self.model.policy.state_meta}")
        log_debug(f"model image keys:{self.model.policy.camera_used}")
        self.model.reset()
        log_info("DexforceVLA Policy loaded.")

        return True

    
    def ready(self):
        if not self.cfg.service:
            if not self._initialize_model():
                return False

        self.shutdown_event.clear()
        self.first_prediction = True
        self.ready_event.set()

        # 记录被拿来进行推理的观测的时间步
        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        if not PolicyServerPureInference._logger_initialized:
            setup_logger()
            PolicyServerPureInference._logger_initialized = True

        log_info(
            f"DexforceVLA Policy server ready at {self.server_host}:{self.server_port}"
        )
        log_info(f"Client timeout: {self.cfg.client_timeout}s")
        log_info(f"Force model reinit: {self.cfg.force_model_reinit}")
        return True
    
    def _handle_control_command(self, request: dict) -> dict:
        try:
            command_type = request.get(ResponseKey.TYPE)
            if command_type == RequestType.SETUP_CONFIG:
                config = request.get(ResponseKey.CONFIG, {})
                success = self.update_config(config)
                if success:
                    self.state = "running"
                    self.error_message = None
                    return {ResponseKey.SUCCESS: True, ResponseKey.STATE: "running"}
                return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: "Failed to update configuration"}
            elif command_type == RequestType.SHUTDOWN:
                self.stop()
                return {ResponseKey.SUCCESS: True, ResponseKey.MESSAGE: "Server shutdown initiated"}
            elif command_type == RequestType.STATUS:
                return {ResponseKey.SUCCESS: True, ResponseKey.STATE: self.state, ResponseKey.ERROR: self.error_message}
            elif command_type == RequestType.STOP:
                # 只重置推理状态，不断开连接不退出主循环
                self.state = "idle"
                self.start_infer_event.clear()
                self.busy_event.clear()
                return {ResponseKey.SUCCESS: True, ResponseKey.STATE: "idle"}
            else:
                return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: f"Unknown command type: {command_type}"}

        except Exception as e:
            log_error(f"Error handling control command: {e}")
            traceback.print_exc()
            self.state = ResponseKey.ERROR
            self.error_message = str(e)
            return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: str(e)}



    def update_config(self, config: Dict) -> bool:
        """更新服务器配置"""
        try:
            self.new_config = config
            self.config_update_pending = True
            log_debug(f"Updating server configuration: {config}")
            # 更新配置
            self.cfg.apply_update(config)
            # 停止当前推理
            self._reset_server_state()
            if self.past_policy_path != self.cfg.pretrained:
                log_info("Reinitializing model with new configuration...")
                self._initialize_model()
                self.past_policy_path = self.cfg.pretrained
            self.model.set_action_horizon(self.cfg.action_horizon)
            self.config_update_pending = False
            log_info("Server configuration updated successfully")
            return True
        except Exception as e:
            log_error(f"Failed to update server config: {e}")
            traceback.print_exc()
            self.config_update_pending = False
            return False



    def get_status(self) -> Dict:
        """获取服务器状态"""
        return {
            "running": self.running,
            "client_connected": self.client_connected,
            "model_loaded": self.model is not None,
            ResponseKey.ERROR: True if log_error.call_count > 0 else False,
            ResponseKey.MESSAGE: self.status_message
        }

    def _update_client_activity(self):
        self.last_client_activity = time.time()

    def _check_client_timeout(self):
        time_since_last_activity = time.time() - self.last_client_activity
        if self.client_connected and time_since_last_activity > self.cfg.client_timeout:
            log_info(
                f"Client timeout detected. No activity for {time_since_last_activity:.1f}s"
            )
            return True
        return False

    def _reset_server_state(self):
        log_info("Resetting server state for new client connection...")
        self.status_message = ''
        # 增加会话ID
        self.current_session_id += 1
        log_info(f"Starting new session #{self.current_session_id}")

        # 清空队列和缓冲区
        while not self.observation_queue.empty():
            try:
                self.observation_queue.get_nowait()
            except Empty:
                break

        while not self.past_obs_queue.empty():
            try:
                self.past_obs_queue.get_nowait()
            except Empty:
                break

        self.current_actions = None
        self.current_timestamp = 0.0
        self.current_timestep = 0

        # 重置事件
        self._predicted_timesteps = set()
        self.actions_ready_event.clear()
        self.busy_event.clear()
        self.start_infer_event.clear()
        self.last_processed_obs = None
        self.first_prediction = True
        self._last_valid_timestep = -1

        self.actionchunks.clear()
        log_error.reset_counter()

        # 重置模型状态
        if self.model is not None:
            try:
                self.Debug_Datasaver = None
                if (
                    self.cfg.save_input
                    or self.cfg.save_vis_images
                    or self.cfg.save_joint_change
                ):
                    self.Debug_Datasaver = Debug_Datasaver(
                        camera_used=self.model.policy.camera_used
                    )
                log_info("Resetting DexforceVLA policy state...")
                self.init_flag = True
                self.model.reset()
                log_info("DexforceVLA policy state reset completed")
            except Exception as e:
                self.status_message = log_error(f"Error resetting DexforceVLA policy: {e}")

        log_info(f"Server state reset completed for session #{self.current_session_id}")
        return True

    
    def _handle_observation(self, request: dict) -> dict:
        try:
            self._update_client_activity()
            self.monitor.record_call()

            if self.model is None:
                self.status_message = log_error(
                    "DexforceVLA Policy is not initialized, cannot process observation"
                )
                return {
                    ResponseKey.STATUS: ResponseKey.ERROR,
                    ResponseKey.MESSAGE: "DexforceVLA Policy not initialized",
                }


            # 获取观测数据
            proprio = request.get(Modality.STATES.value, [])
            timestamp = request.get(CommonKey.TIMESTAMP.value, time.time())
            client_timestep = request.get(CommonKey.TIMESTEP.value, 0)
            must_go = request.get(CommonKey.MUST_GO.value, False)
            start_infer = request.get("start_infer", False)
            head_size = request.get("head_target_size", (640, 360))
            hand_size = request.get("hand_target_size", (640, 360))
            self.end_effector_limit = request.get(CommonKey.END_EFFECTOR_LIMIT.value, [0, 100])
            instruction = request.get(CommonKey.INSTRUCTION.value,"")
            self.inference_latency = request.get("time_infer", 0.05)
            images = {}
            image_sizes = ((head_size[1], head_size[0], 3), 
                           (head_size[1], head_size[0], 3), 
                           (hand_size[1], hand_size[0], 3), 
                           (hand_size[1], hand_size[0], 3))
            
            # 解压缩所有图像
            for image_keys, image_size in zip(self.model.policy.camera_used, image_sizes):
                images[image_keys] = np.frombuffer(request.get(image_keys), dtype=np.uint8).reshape(image_size)



            if start_infer:
                self.start_infer_event.set()
            else:
                self.start_infer_event.clear()



            # 创建观测对象
            obs = {
                Modality.IMAGES.value: {},
                Modality.STATES.value: proprio,
                CommonKey.DISPS.value: [None],
                CommonKey.INSTRUCTION.value: instruction

            }
            for image_keys in self.model.policy.camera_used:
                obs[Modality.IMAGES.value][image_keys] = images[image_keys]

            timed_observation = TimedObservation(
                timestamp, client_timestep, obs, must_go
            )
            obs_timestep = timed_observation.get_timestep()

            if self.start_infer_event.is_set():
                if not self._enqueue_observation(timed_observation):
                    log_info(f"Observation #{obs_timestep} has been filtered out")
            else:
                # 没有触发推理, 动作正常执行, 更新模型里面的历史
                # 处理推理的时候错过的那些历史数据
                while not self.past_obs_queue.empty():
                    observation_t = self.past_obs_queue.get()
                    obs_past = self._map_obs_to_model(observation_t)
                    _ = self.model.from_real_obs(obs_past)
                    log_info(f"清理推理的时候错过的状态缓存 #{observation_t.get_timestep()}")

                obs = self._map_obs_to_model(timed_observation)
                with self.observation_queue_lock:
                    _ = self.model.from_real_obs(obs)


            return {
                ResponseKey.STATUS: "received",
                CommonKey.TIMESTEP.value: client_timestep,
                "session_id": self.current_session_id,
            }
            
        except Exception as e:
            self.state = ResponseKey.ERROR                                   
            self.error_message = str(e)                           
            self.status_message = log_error(f"Error handling observation: {e}")
            traceback.print_exc()
            return {ResponseKey.STATUS: ResponseKey.ERROR, ResponseKey.MESSAGE: str(e)}
        

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """接收到的观测入队列"""
        if (
            obs.must_go  # must_go标志，第一次观测时候会被设置为真，以及动作聚合结束后，动作队列都被执行完了也会为真
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = (
                self.last_processed_obs.get_timestep()
                if self.last_processed_obs
                else "None"
            )
            log_info(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs} | This processed obs: {obs.get_timestep()}"
            )

            if self.observation_queue.full():

                _ = self.observation_queue.get_nowait()
                log_info("Observation queue was full, removed oldest observation.")

            self.observation_queue.put(obs)
            return True
        else:
            return False

    def _obs_sanity_checks(
        self, obs: TimedObservation, previous_obs: TimedObservation
    ) -> bool:
        """观测相过滤检查, 可以自定义, 如果相似则返回False"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            log_info(
                f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!"
            )
            return False

        elif self.observations_similar(obs, previous_obs):
            log_info(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        elif self.busy_event.is_set():
            log_info(f"推理中,该观测入队缓存 #{obs.get_timestep()}")
            self.past_obs_queue.put(deepcopy(obs))
            return False

        else:
            return True

    def observations_similar(
        self, obs: TimedObservation, previous_obs: TimedObservation
    ) -> bool:
        # 对于DexforceVLA，我们主要依赖时间戳对齐，这里返回False总是处理新观测
        return False

    def inference_actions(self, thread_stop_event):
        log_info("DexforceVLA Inference thread started")

        while self.running and not thread_stop_event.is_set():
            try:
                # 检查客户端连接状态
                if not self.client_connected:
                    time.sleep(0.5)
                    continue

                # 检查模型是否可用
                if self.model is None:
                    if not self._model_none_warned:
                        log_warning("DexforceVLA Policy is not initialized, cannot perform inference")
                        self._model_none_warned = True
                    time.sleep(0.5)
                    continue

                if self.start_infer_event.is_set() and not self.busy_event.is_set():

                    # 获取新观测
                    try:
                        getactions_st = time.perf_counter()

                        obs = self.observation_queue.get(timeout=0.01)

                        getactions_ = time.perf_counter()
                        with self._predicted_timesteps_lock:
                            self._predicted_timesteps.add(obs.get_timestep())

                        log_debug(
                            f"observation_queue.get {(getactions_-getactions_st)*1000:.2f}ms"
                        )
                    except Empty:
                        continue

                    log_info(
                        f"Session #{self.current_session_id} - Running DexforceVLA inference for observation #{obs.get_timestep()}"
                    )

                    action_chunk = self._predict_action_chunk(obs)
                    time.sleep(
                        max(
                            0,
                            self.inference_latency
                            - max(0, time.perf_counter() - getactions_st),
                        )
                    )
                    self._send_actions(
                        action_chunk,
                        obs.get_timestamp(),
                        obs.get_timestep() + 1,
                    )

                    self.busy_event.clear()
                    self.start_infer_event.clear()

            except Empty:
                self.actions_ready_event.clear()
            except Exception as e:
                self.state = ResponseKey.ERROR                               
                self.error_message = str(e)                      
                self.status_message = log_error(f"Error in DexforceVLA inference actions: {e}")
                time.sleep(0.1)


    def _map_obs_to_model(self, observation_t: TimedObservation):
        observation = observation_t.get_observation()
        # 根据数据类型决定乘数因子
        multiplier = 1.0 if self.cfg.data_type == "real" else self.cfg.hand_sim_max_limit
        for type_idx in self.gripper_types_to_process:
            proprio_type = SUPPORTED_PROPRIO_TYPES[type_idx]
            if proprio_type in observation[Modality.STATES.value]:
                if proprio_type == SUPPORTED_PROPRIO_TYPES[4] or proprio_type == SUPPORTED_PROPRIO_TYPES[5]:
                    # 归一化： 从观测的手部实际范围[0,100]到模型需求的范围[0,1]* self.cfg.hand_sim_max_limit
                    observation[Modality.STATES.value][proprio_type] = (
                        normalize_array(
                            observation[Modality.STATES.value][proprio_type],
                            src_min=self.end_effector_limit[0],
                            src_max=self.end_effector_limit[1],
                            dst_min=self.cfg.model_end_effector_limit[0],
                            dst_max=self.cfg.model_end_effector_limit[1],
                            reverse=self.cfg.inverse_gripper
                        ) * multiplier
                    ).astype(np.float32)


                if proprio_type == SUPPORTED_PROPRIO_TYPES[6] or proprio_type == SUPPORTED_PROPRIO_TYPES[7]:
                    # 归一化： 从观测的二指夹实际范围[0,100]到模型需求的范围[0,1]
                    observation[Modality.STATES.value][proprio_type] = normalize_array(
                        observation[Modality.STATES.value][proprio_type],
                        src_min=self.end_effector_limit[0],
                        src_max=self.end_effector_limit[1],
                        dst_min=self.cfg.model_end_effector_limit[0],
                        dst_max=self.cfg.model_end_effector_limit[1],
                        reverse=self.cfg.inverse_gripper
                    )

        states_keys = list(observation[Modality.STATES.value].keys())
        for key in states_keys:

            if key not in self.model.policy.state_meta:
                _ = observation[Modality.STATES.value].pop(key)
            
            
        obs = {
            Modality.IMAGES.value: [
                observation[Modality.IMAGES.value][cam_name]
                for cam_name in self.model.policy.camera_used
            ],
            Modality.STATES.value: observation[Modality.STATES.value],
            CommonKey.DISPS.value: observation[CommonKey.DISPS.value],
            CommonKey.INSTRUCTION.value: observation[CommonKey.INSTRUCTION.value]
        }

        return obs

    def _predict_action_chunk(self, observation_t: TimedObservation):
        self.busy_event.set()

        log_info(
            f"Session #{self.current_session_id} - Predicting DexforceVLA action chunk for observation #{observation_t.get_timestep()}"
        )

        self.last_processed_obs: TimedObservation = observation_t

        batch = self._map_obs_to_model(observation_t)
        with self.current_actions_lock:
            batch_map = self.model.from_real_obs(batch, init=self.init_flag)
        self.init_flag = False


        if batch is None:
            self.status_message = log_error("Failed to build DexforceVLA observation")

        try:
            with torch.no_grad():
                start_predict = time.perf_counter()
                if batch[CommonKey.INSTRUCTION.value] != "":
                    batch_map[Modality.LANG.value] = batch[CommonKey.INSTRUCTION.value]


                action_chunk = self.model.step(**batch_map)
                finish_predict = (time.perf_counter() - start_predict) * 1000


                if self.Debug_Datasaver is not None:
                    if self.cfg.save_input:
                        try:
                            self.Debug_Datasaver.save_batch(
                                images=batch[Modality.IMAGES.value],
                                states_value=batch[Modality.STATES.value],
                                prompt_value=batch[CommonKey.INSTRUCTION.value] if CommonKey.INSTRUCTION.value in batch else "",
                            )
                        except Exception as e:
                            log_error(f"Error saving input data for debugging: {e}")
                    if self.cfg.save_vis_images:
                        self.Debug_Datasaver._save_vis_images(self.model.vis_images)
                    if self.cfg.save_joint_change:
                        # 传入历史 pickle 路径做对比
                        self.Debug_Datasaver.plot_joint_changes(
                            action_chunk[JointType.QPOS.value],
                            prev_pickle_path=self.cfg.prev_pickle,
                        )
                        # 保存本次 action_chunk 为 pickle，以便后续对比分析使用
                        self.Debug_Datasaver._save_action_chunk_pkl(action_chunk[JointType.QPOS.value])

                if self.cfg.use_smooth:
                    for key in action_chunk[JointType.QPOS.value].keys():
                        action_chunk[JointType.QPOS.value][key] = gaussian_filter1d(
                            action_chunk[JointType.QPOS.value][key],
                            sigma=2,
                            axis=0,
                            mode="nearest",
                        )



                # NOTE：对模型输出的结果->[end_effector_limit[0],end_effector_limit[1]]，保存正确的限制范围的关节值
                # 4 5 hand 6 7 pripper
                multiplier = 1.0 if self.cfg.data_type == "real" else self.cfg.hand_sim_max_limit
                for type_idx in self.gripper_types_to_process:
                    proprio_type = SUPPORTED_PROPRIO_TYPES[type_idx]
                    if proprio_type in action_chunk[JointType.QPOS.value].keys():
                        for action in action_chunk[JointType.QPOS.value][proprio_type]:
                            if proprio_type == SUPPORTED_PROPRIO_TYPES[4] or proprio_type == SUPPORTED_PROPRIO_TYPES[5]:
                                normalized = normalize_array(
                                    action,
                                    src_min=self.cfg.model_end_effector_limit[0], 
                                    src_max=self.cfg.model_end_effector_limit[1], 
                                    dst_min=self.end_effector_limit[0], 
                                    dst_max=self.end_effector_limit[1],
                                    reverse=self.cfg.inverse_gripper
                                ) * (1 / multiplier) 
                                action[:] = normalized[:] 

                            elif proprio_type == SUPPORTED_PROPRIO_TYPES[6] or proprio_type == SUPPORTED_PROPRIO_TYPES[7]:
                                normalized = normalize_array(
                                    action,
                                    src_min=self.cfg.model_end_effector_limit[0], 
                                    src_max=self.cfg.model_end_effector_limit[1], 
                                    dst_min=self.end_effector_limit[0], 
                                    dst_max=self.end_effector_limit[1], 
                                    reverse=self.cfg.inverse_gripper
                                )
                                action[:] = normalized[:]



                if self.Debug_Datasaver:
                    self.Debug_Datasaver.update_save_path()

                self.step_time.append(finish_predict)

                avg_time = np.mean(self.step_time)
                log_debug(f"Step 时间 : {finish_predict:.2f}ms")
                log_debug(f"DexforceVLA 平均推理时间: {avg_time:.2f}ms")

        except Exception as e:
            traceback.print_exc()
            self.status_message = log_error(f"Error in DexforceVLA policy.select_action: {e}")

        return action_chunk

    def _send_actions(self, action_chunk: dict, timestamp: float, timestep: int):
        # 创建元数据
        metadata = {
            "timestamp": timestamp,
            "timestep": timestep,
            "session_id": self.current_session_id,
        }

        while not self.actions_queue.empty():
            try:
                self.actions_queue.get_nowait()
            except Empty:
                break

        while not self.actions_metadata_queue.empty():
            try:
                self.actions_metadata_queue.get_nowait()
            except Empty:
                break

        self.actions_queue.put(action_chunk)
        self.actions_metadata_queue.put(metadata)

        self.actions_ready_event.set()

        log_info(
            f"Session #{self.current_session_id} - DexforceVLA Actions ready for timestep {timestep}"
        )

    def _handle_get_actions(self, request: dict) -> dict:
        try:
            self._update_client_activity()

            if not self.actions_ready_event.is_set():
                return {
                    ResponseKey.STATUS: "no_actions",
                    ResponseKey.MESSAGE: "No DexforceVLA actions available",
                }

            try:
                action_chunk = self.actions_queue.get_nowait()
                metadata = self.actions_metadata_queue.get_nowait()

                response = {
                    ResponseKey.STATUS: ResponseKey.SUCCESS,
                    Modality.ACTIONS.value: action_chunk,
                    CommonKey.TIMESTAMP.value: metadata["timestamp"],
                    CommonKey.TIMESTEP.value: metadata["timestep"],
                    "session_id": metadata["session_id"],
                }

                log_info(
                    f"Session #{self.current_session_id} - Returning DexforceVLA actions for timestep {metadata['timestep']}"
                )

                if self.actions_queue.empty():
                    self.actions_ready_event.clear()
                return response

            except Empty:
                self.actions_ready_event.clear()
                return {
                    ResponseKey.STATUS: "no_actions",
                    ResponseKey.MESSAGE: "No DexforceVLA actions available",
                }

        except Exception as e:
            self.state = ResponseKey.ERROR                                    
            self.error_message = str(e)                             
            self.status_message = log_error(f"Error handling get_actions: {e}")
            return {ResponseKey.STATUS: ResponseKey.ERROR, ResponseKey.MESSAGE: str(e)}
        
    def _client_connection_monitor(self, thread_stop_event):
        log_info("Client connection monitor started")

        while self.running and not thread_stop_event.is_set():
            try:
                # 检查客户端是否连接
                if not self.client_connected:
                    log_info(
                        f"Client not connected. Waiting for client to connect..."
                    )
                    # 主动断开超时的客户端连接
                    self.network_server.close()

            except Exception as e:
                self.status_message = log_error(f"Error in client connection monitor: {e}")
            time.sleep(5)


    def _on_client_disconnected(self):
        """网络服务器检测到客户端断开连接时的回调"""
        log_info("Network server detected client disconnection")
        self.client_connected = False
        
        # 如果设置了断开连接回调，则调用
        if self.on_client_disconnected_callback:
            self.on_client_disconnected_callback()


    def monitor_thread(self):
        while True:
            stats = self.monitor.get_stats()
            log_info(f"频率: {stats['frequency']:.2f} 次/秒, "
                f"窗口内调用: {stats['calls_in_window']}")
            time.sleep(1)

    def start(self):
        if self.ready():
            thread_stop_event = threading.Event()
            while self.running:
                try:
                    if self.network_server.start():
                        log_info(f"DexforceVLA Policy server started on {self.server_host}:{self.server_port}")
                        self.client_connected = True
                        self.last_client_activity = time.time()
                        client_thread = threading.Thread(
                            target=self._client_connection_monitor,
                            args=(thread_stop_event,), daemon=True)
                        inference_thread = threading.Thread(
                            target=self.inference_actions,
                            args=(thread_stop_event,), daemon=True)
                        client_thread.start()
                        inference_thread.start()
                        log_info(f"Client connected, starting session #{self.current_session_id}")
                        try:
                            self.network_server.handle_requests(
                                lambda: self.running and self.client_connected)
                        except Exception as e:
                            self.status_message = log_error(f"Error in request handling: {e}")
                        log_info(f"Client disconnected for session #{self.current_session_id}")
                        self.client_connected = False
                        self.network_server.close()
                        thread_stop_event.set()
                        client_thread.join(timeout=1)
                        inference_thread.join(timeout=1)
                        thread_stop_event.clear()
                        if not self.cfg.service:
                            self._reset_server_state()
                        log_info(f"Session completed, ready for new client")
                    else:
                        self.status_message = log_error("Failed to start network server, retrying...")
                        time.sleep(5.0)
                except Exception as e:
                    self.status_message = log_error(f"Error in server main loop: {e}")
                    self.client_connected = False
                    self.network_server.close()
                    time.sleep(5.0)
        else:
            self.status_message = log_error("Failed to initialize DexforceVLA model")


            
    def stop(self):
        """停止服务器"""
        log_info("Stopping DexforceVLA policy server...")

        # 设置关闭事件
        self.shutdown_event.set()

        self.client_connected = False

        # 关闭网络服务器
        self.network_server.close()
        # self.tracer.stop()
        # self.tracer.save()

        log_info("DexforceVLA Policy server stopped successfully")

    def __del__(self):
        if not self.shutdown_event.is_set():
            self.stop()
