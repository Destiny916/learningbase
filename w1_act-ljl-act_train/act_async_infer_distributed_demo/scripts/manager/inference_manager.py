from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
from typing import Optional
from rclpy.timer import Timer

from std_srvs.srv import Trigger
from inference_interfaces.srv import StartInference, Deploy, GetModelInfo
from inference_interfaces.msg import InferenceStatus

from act_async_infer_distributed_demo.scripts.manager.config_registry import ConfigRegistry
from act_async_infer_distributed_demo.scripts.manager.inference_client_controller import InferenceClientController

from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_info,
    log_error,
    log_warning
)
from act_async_infer_distributed_demo.scripts.inference_config import ResponseKey, RequestType


class InferenceManager(Node):
    def __init__(self, args):
        super().__init__('inference_manager')
        
        self.config_registry = ConfigRegistry()
        self.client_controller = InferenceClientController(args.client_host,args.client_port)
        
        self.current_status = InferenceStatus()
        self.current_status.status = InferenceStatus.IDLE
        self.current_status.message = ''
        self.last_client_status = InferenceStatus()
        self.last_client_status.status = InferenceStatus.IDLE
        self.startup_config = None
        self.current_config = None
        
        self.setupping_config = False
        self.lock = threading.RLock()
        
        self.cb_group = ReentrantCallbackGroup()
        
        self._init_services()
        
        self.status_publisher = self.create_publisher(
            InferenceStatus,
            '/inference/status',
            10
        )
        
        self.status_timer = self.create_timer(1.0, self._publish_status)
        
        self.status_mapping = {
            "idle": InferenceStatus.IDLE,
            "running": InferenceStatus.RUNNING,
            "error": InferenceStatus.ERROR,
        }

        # 监控xwiz_ros2_bridge节点的看门狗
        self._target_node = "/xwiz_ros2_bridge"
        self._node_alive = False
        self._monitoring = False
        self._watchdog_timer: Optional[Timer] = None

        # Trigger 客户端
        self._stop_client = self.create_client(
            Trigger,
            '/inference/stop_inference'
        )
        self._init_watchdog()
        
        
        
        log_info("InferenceManager initialized")
        

    
    def _init_services(self):
        """初始化ROS2服务"""
        # StartInference服务
        self.start_service = self.create_service(
            StartInference,
            '/inference/start_inference',
            self._handle_start_inference,
            callback_group=self.cb_group
        )
        
        # StopInference服务（使用Trigger接口）
        self.stop_service = self.create_service(
            Trigger,
            '/inference/stop_inference',
            self._handle_stop_inference,
            callback_group=self.cb_group
        )
    
        self.start_service = self.create_service(
            GetModelInfo,
            '/inference/get_model_info',
            self._handle_get_model_info,
            callback_group=self.cb_group
        )
        
        # Deploy服务
        self.deploy_service = self.create_service(
            Deploy,
            '/inference/deploy',
            self._handle_deploy,
            callback_group=self.cb_group
        )
    

    def _load_startup_config(self):
        """加载启动默认配置"""
        try:
            selection = self.config_registry.load_startup_selection()
            model_id = selection.get('startup_model_id', 0)
            task_id = selection.get('startup_task_id', 0)
            
            # 使用默认模式（仿真）
            self.startup_config = self.config_registry.resolve_config(
                model_id, task_id, mode=1
            )
            self.current_config = self.startup_config
            
            log_info(
                f"Loaded startup config: model_id={model_id}, task_id={task_id}"
            )
        except Exception as e:
            log_error(f"Failed to load startup config: {e}")
            self.startup_config = None
            self.current_config = None
    
    
    def _update_status_from_client(self):
        try:
            client_status = self.client_controller.get_status()
            if not client_status.get(ResponseKey.SUCCESS, False):
                self.current_status.status = InferenceStatus.ERROR
                self.current_status.message = "[manager] 推理client不在线"
                return

            status = client_status.get(ResponseKey.STATUS, {})
            state = status.get(ResponseKey.STATE, "idle")
            if state in self.status_mapping:
                self.current_status.status = self.status_mapping[state]
            self.current_status.message = "ok" if not status.get(ResponseKey.ERROR) else status.get(ResponseKey.ERROR)
            self.last_client_status = self.current_status

        except Exception as e:
            self.current_status.status = InferenceStatus.ERROR
            self.current_status.message = f"[manager] {e}"



    def _handle_start_inference(self, request, response):
        """处理开始推理请求"""
        with self.lock:
            try:
                log_info(
                    f"Received start_inference: mode={request.mode}, "
                    f"model={request.model}, task={request.task}"
                )
                
                # 检查当前状态（从客户端获取最新状态）
                self._update_status_from_client()
                if self.current_status.status == InferenceStatus.RUNNING:
                    response.success = False
                    response.message = "Inference is already running"
                    return response
                
                # 解析配置
                client_config, server_config = self.config_registry.resolve_config(
                    request.model, request.task, request.mode
                )
                
                # 通过客户端控制器下发配置
                self.setupping_config = True
                if not self.client_controller.setup_config(client_config, server_config):
                    self._update_status_from_client()
                    response.success = False
                    response.message = self.current_status.message or "[manager] 推理服务启动失败"
                    log_error("推理服务启动失败")
                    return response
                self.setupping_config = False
                # 更新当前配置
                self.current_config = (client_config, server_config)
                
                # 再次从客户端获取状态以确保准确
                self._update_status_from_client()
                
                response.success = True
                response.message = "推理服务启动成功"
                log_info("Inference started")
                
            except Exception as e:
                response.success = False
                response.message = f"推理服务启动失败: {str(e)}"
                
                # 设置错误状态
                self.current_status.status = InferenceStatus.ERROR
                self.current_status.message = response.message
                log_error(response.message)
            return response


    def _handle_get_model_info(self, request, response):
        """处理开始推理请求"""
        with self.lock:
            try:
                log_info(
                    f"Received get_model_info: model={request.model}"
                )
                # 解析配置
                camera_used, joint_groups = self.config_registry.resolve_state_meta(request.model)

                response.selected_cameras = camera_used
                response.joint_groups = joint_groups
                log_info(f"joint_groups:{joint_groups}")


                log_info("Get model info successfully")
            except Exception as e:
                response.selected_cameras = []
                response.joint_groups = []                
                # 设置错误状态
                self.current_status.status = InferenceStatus.ERROR
                self.current_status.message = f"Failed to get model info: {str(e)}"
                log_error(self.current_status.message)
            return response




    
    def _handle_stop_inference(self, request, response):
        """处理停止推理请求"""
        with self.lock:
            try:
                log_info("Received stop_inference request")
                
                # 停止客户端
                self.client_controller.stop()

                response.success = True
                response.message = "Inference stopped successfully"
                
                log_info("Inference stopped")
                
            except Exception as e:
                response.success = False
                response.message = f"推理服务停止失败: {str(e)}"
                
                # 设置错误状态
                self.current_status.status = InferenceStatus.ERROR
                self.current_status.message = response.message
                log_error(response.message)
            
            return response
    
    def _handle_deploy(self, request, response):
        """处理部署请求"""
        with self.lock:
            try:
                log_info("Received deploy request")
                
                # 刷新启动配置
                self._load_startup_config()
                
                # 重置当前配置为启动配置
                if self.startup_config:
                    client_config, server_config = self.startup_config
                    self.client_controller.setup_config(client_config, server_config)
                    self.current_config = self.startup_config
                
                response.success = True
                response.message = "Configuration deployed successfully"
                
                log_info("Configuration deployed")
                
            except Exception as e:
                log_error(f"Failed to deploy: {e}")
                response.success = False
                response.message = f"Failed to deploy: {str(e)}"
            
            return response
    
    def _publish_status(self):
        """定时发布状态"""
        if not self.setupping_config:
            self._update_status_from_client()
            self.status_publisher.publish(self.current_status)
        else:
            self.status_publisher.publish(self.last_client_status)
    
    def _init_watchdog(self) -> None:
        self._watchdog_timer = self.create_timer(1.0, self._watchdog_callback)
        log_info(f"看门狗已启动, 等待 {self._target_node} 上线...")
    
    def _watchdog_callback(self) -> None:
        alive = self._node_is_present(self._target_node)
        if not self._monitoring:
            if alive:
                self._monitoring = True
                self._node_alive = True
                log_info(f"看门狗: {self._target_node} 已上线, 进入持续监控")
        else:
            if not alive and self._node_alive:
                self._node_alive = False
                log_warning(f"看门狗: {self._target_node} 已下线, 触发 /inference/stop_inference")
                self._trigger_stop_service()
            elif alive and not self._node_alive:
                self._node_alive = True
                log_info(f"看门狗: {self._target_node} 已重新上线")

    def _trigger_stop_service(self) -> None:
        """通过 Trigger 客户端调用 /inference/stop_inference"""
        if not self._stop_client.service_is_ready():
            log_error("看门狗: /inference/stop_inference 服务不可用")
            return
        req = Trigger.Request()
        future = self._stop_client.call_async(req)
        future.add_done_callback(self._on_stop_response)

    def _on_stop_response(self, future) -> None:
        """处理停止服务的异步响应"""
        try:
            resp = future.result()
            if resp.success:
                log_info(f"看门狗: 停止成功 — {resp.message}")
            else:
                log_error(f"看门狗: 停止失败 — {resp.message}")
        except Exception as e:
            log_error(f"看门狗: 停止服务调用异常 — {e}")

    def _node_is_present(self, node_name: str) -> bool:
        for name, ns in self.get_node_names_and_namespaces():
            full = f"{ns}/{name}" if ns != "/" else f"/{name}"
            if full == node_name or f"/{name}" == node_name or name == node_name.lstrip("/"):
                return True
        return False

    
    def destroy(self):
        """清理资源"""
        self.client_controller.shutdown()
        super().destroy_node()
