from pathlib import Path

import yaml


CONFIG_PATH = (
    Path(__file__).parents[2]
    / "w1_act-ljl-act_train"
    / "xwiz_real_runtime"
    / "pc2_ros2_bridge.yaml"
)
RUNTIME_DIR = CONFIG_PATH.parent
CLIENT_CONFIG_PATH = RUNTIME_DIR / "client_runtime.json"
ROS_ENV_PATH = RUNTIME_DIR / "pc2_ros2.env"
SYSTEMD_DIR = RUNTIME_DIR / "systemd"


def test_inference_services_use_explicit_request_response_message_types():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    outbound = config["outbound_types"]
    inbound = config["inbound_types"]
    services = {
        "start_inference": "inference_interfaces/srv/StartInference",
        "stop_inference": "std_srvs/srv/Trigger",
        "deploy": "inference_interfaces/srv/Deploy",
        "get_model_info": "inference_interfaces/srv/GetModelInfo",
        "export_shadow_segments": "inference_interfaces/srv/ExportShadowSegments",
    }

    for name, base_type in services.items():
        event = f"service_/inference/{name}"
        assert outbound[event] == f"{base_type}_Request"
        assert inbound[f"{event}/response"] == f"{base_type}_Response"


def test_pc2_xwiz_loads_the_w1_ros_overlay():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["ament"]["overlays"] == ["/home/dexforce/w1/install"]


def test_pc2_xwiz_subscribes_to_both_head_resize_topics():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    inbound = config["inbound_types"]
    assert inbound["/camera/left_eye_resize"] == "sensor_msgs/msg/Image"
    assert inbound["/camera/right_eye_resize"] == "sensor_msgs/msg/Image"


def test_pc2_xwiz_uses_cyclonedds_on_the_pc2_robot_interface():
    env = ROS_ENV_PATH.read_text(encoding="utf-8")
    assert "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in env
    assert "NetworkInterface" in env
    assert "192.168.20.21" in env
    assert "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" not in env


def test_real_client_uses_the_physical_wrist_camera_topics():
    config = yaml.safe_load(CLIENT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["cam_hand_left_topic"] == "/camera_l/color/image_rect_raw"
    assert config["cam_hand_right_topic"] == "/camera_r/color/image_rect_raw"


def test_real_client_service_does_not_depend_on_black_wrist_publishers():
    unit = (SYSTEMD_DIR / "xwiz-real-client.service").read_text(encoding="utf-8")
    assert "xwiz-black-wrist.service" not in unit


def test_pc1_kfc_v1_service_publishes_resize_images_on_domain_20():
    unit = (SYSTEMD_DIR / "xwiz-kfc-v1.service").read_text(encoding="utf-8")
    assert "ROS_DOMAIN_ID=20" in unit
    assert "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in unit
    assert "dexe_sensors_launch kfc_nodes.launch.py" in unit
    assert "kfc_mode:=resize_compressed" in unit
