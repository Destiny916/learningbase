import json
import os
from pathlib import Path
from typing import Dict, Tuple
from act_async_infer_distributed_demo.scripts.utils_distributed import load_json, find_model_dirs, log_error
from act_async_infer_distributed_demo.scripts.w1_mapping import (
    w1qpos_names_map, 
    camera_used_names_map, 
    w1qpos_group_map,
    ModelNameKeys
)
from inference_interfaces.msg import JointGroup


class ConfigRegistry:
    def __init__(self):
        self.config_base_path = Path.home() / "workspace" /".dexforce" / "XWiz" / "model_deployments"
    def load_startup_selection(self) -> Dict:
        """加载启动默认配置"""
        selection_file = self.config_base_path / "inference_state.json"
        if selection_file.exists():
            with open(selection_file, 'r') as f:
                return json.load(f)
        return {"startup_model_id": 0, "startup_task_id": 1}

    def resolve_config(self, model_id: int, task_id: int, mode: int) -> Tuple[Dict, Dict]:
        """根据模型ID、任务ID和模式解析配置

        Args:
            model_id: 模型ID
            task_id: 任务ID
            mode: 1=仿真，2=真机
            
        Returns:
            Tuple[client_config, server_config]
        """
        # 根据模式确定机器人类型
        robot_type = "simulation" if mode == 1 else "real_robot"
        

        
        config_json_path = f"tasks/{task_id}/task_config.json"
        

        pretrained_path = self.find_pretrained_path(model_id)
        if not pretrained_path:
            log_error(f"未找到对应路径{pretrained_path}下的模型")
            return {}, {}
        
        config_json = load_json(f"{self.config_base_path}/{config_json_path}")
        # 客户端更新参数
        client_config = config_json.get("client_config", {})
        max_duration = config_json.get("max_duration", 60)
        control_freq = client_config.get("collect_frequency", 8)
        max_steps = max_duration * control_freq
        model_name = self.get_model_name(model_id)
        
        if robot_type == "simulation":
            client_config.update({"home_position": ""})
        client_config.update({"mode": mode})
        client_config.update({"prompt": config_json.get("instruction", client_config.get("prompt", ""))})
        client_config.update({"max_steps": max_steps})

        # 服务器更新参数
        server_config = config_json.get("server_config", {})
        server_config.update({"pretrained": pretrained_path})
        server_config.update({"action_horizon": client_config.get("action_horizon", 32)})
        server_config.update({"model_name": model_name})
        
        return client_config, server_config
    
    def find_pretrained_path(self, model_id: int) -> str:
        """根据模型ID和任务ID查找预训练模型路径"""
        model_path = f"{self.config_base_path}/models/{model_id}"
        pretrained_paths = find_model_dirs(model_path)
        if not pretrained_paths:
            log_error(f"未找到对应路径{model_path}下的模型")
        return pretrained_paths[0] if pretrained_paths else ""
        

    def resolve_state_meta(self, model_id: int) -> Dict:
        """解析状态元配置"""
        state_meta_path = self.find_state_meta_path(model_id)

        state_config = load_json(state_meta_path)
        camera_used = state_config.get("camera_used", [])
        state_meta = state_config.get("state_meta", {})
        state_meta_output = state_config.get("output", [])
        from act_async_infer_distributed_demo.scripts.utils_distributed import log_info

        # 取state_meta和state_meta_output的交集
        state_meta_filtered = list(set(state_meta) | set(state_meta_output))

        camera_used = [camera_used_names_map[name] for name in camera_used if name in camera_used_names_map]

        joint_groups = []

        for key in w1qpos_group_map:
            entry = JointGroup()
            group_name = w1qpos_group_map[key][0]
            joint_names = w1qpos_names_map[key]
            entry.group_name = group_name
            entry.joints = joint_names

            if key in state_meta_filtered:
                entry.is_selected = True
            else:
                entry.is_selected = False

            joint_groups.append(entry)
        
                
        
        return camera_used, joint_groups
    
    def find_state_meta_path(self, model_id: int) -> str:
        """根据模型ID查找状态元配置路径"""
        pretrained_path = self.find_pretrained_path(model_id)
        state_meta_path = os.path.join(pretrained_path, "config.json")
        if not os.path.exists(state_meta_path):
            log_error(f"未找到对应路径{state_meta_path}下的状态元配置")
            return ""
        return state_meta_path
    
    def get_model_name(self, model_id: int) -> str:
        """根据模型ID获取模型名称"""
        state_meta_path = self.find_state_meta_path(model_id)
        state_config = load_json(state_meta_path)
        
        brain_name = ""
        cerebellum_name = ""
        
        if ModelNameKeys.VISION_LANGUAGE.value in state_config.get(ModelNameKeys.BRAIN.value, {}).get(ModelNameKeys.ENCODERS.value, {}):
            brain_name = ModelNameKeys.VISION_LANGUAGE.value
            brain_name = state_config.get(ModelNameKeys.BRAIN.value, {}).get(ModelNameKeys.ENCODERS.value, {}).get(ModelNameKeys.VISION_LANGUAGE.value, {}).get("name", "")

        elif ModelNameKeys.IMAGES.value in state_config.get(ModelNameKeys.BRAIN.value, {}).get(ModelNameKeys.ENCODERS.value, {}):
            brain_name = state_config.get(ModelNameKeys.BRAIN.value, {}).get(ModelNameKeys.ENCODERS.value, {}).get(ModelNameKeys.IMAGES.value, {}).get("name", "")
            
        cerebellum_name = state_config.get(ModelNameKeys.CEREBELLUM.value, {}).get("name", "")
        
        if brain_name or cerebellum_name:
            return "{}_{}".format(brain_name,cerebellum_name)
        else:
            return "unknown"