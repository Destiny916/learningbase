from enum import Enum

origin_joint_names = [
    "ANKLE",
    "KNEE",
    "BUTTOCK",
    "WAIST",
    "NECK1",
    "NECK2",
    "LEFT_J1",
    "LEFT_J2",
    "LEFT_J3",
    "LEFT_J4",
    "LEFT_J5",
    "LEFT_J6",
    "LEFT_J7",
    "RIGHT_J1",
    "RIGHT_J2",
    "RIGHT_J3",
    "RIGHT_J4",
    "RIGHT_J5",
    "RIGHT_J6",
    "RIGHT_J7",
]

hand_joint_names = [
    "T_CMC_YAW",
    "T_MCP",
    "IF_MCP_PITCH",
    "MF_MCP_PITCH",
    "RF_MCP_PITCH",
    "LF_MCP_PITCH",
]

ordered_body_names = [
    "LEFT_J1",
    "LEFT_J2",
    "LEFT_J3",
    "LEFT_J4",
    "LEFT_J5",
    "LEFT_J6",
    "LEFT_J7",
    "RIGHT_J1",
    "RIGHT_J2",
    "RIGHT_J3",
    "RIGHT_J4",
    "RIGHT_J5",
    "RIGHT_J6",
    "RIGHT_J7",
]
hand_gestures = {
    "normal": [0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    "normal2": [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
    "cup": [0.0, 100.0, 35.0, 45.0, 47.0, 37.0],
    "pinch": [65.0, 100.0, 70.0, 75.0, 100.0, 100.0],
    "fist": [100.0, 30.0, 100.0, 100.0, 100.0, 100.0],
    "like": [0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
    "heart": [0.0, 100.0, 60.0, 70.0, 60.0, 60.0],
    "bull": [90.0, 80.0, 0.0, 100.0, 100.0, 0.0],
    "gun": [0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
    "six": [0.0, 0.0, 100.0, 100.0, 100.0, 0.0],
    "one": [100.0, 70.0, 0.0, 100.0, 100.0, 100.0],
    "salute": [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ok": [60.0, 90.0, 60.0, 0.0, 0.0, 0.0],
}


class CommonKey(Enum):
    """通用数据字典的键名枚举"""

    DISPS = "disps"
    TIMESTAMP = "timestamp"
    TIMESTEP = "timestep"
    MUST_GO = "must_go"
    SHAPE = "shape"
    SIZE = "size"
    LEFT = "left"
    RIGHT = "right"
    IMAGES = "images"
    STATES = "states"
    ACTION = "action"
    END_EFFECTOR_LIMIT = "end_effector_limit"
    INSTRUCTION = "instruction"

class ImageKey(Enum):
    """图像数据字典的键名枚举"""

    CAM_HIGH_R = "cam_high_r"
    CAM_HAND_LEFT = "cam_left_wrist"
    CAM_HAND_RIGHT = "cam_right_wrist"
    FRONT_IMG = "front_img"
    CAM_HIGH = "cam_high"

class EnumIndex(Enum):
    gripper_types_to_process = [4, 5, 6, 7]

class JointNamesKey(Enum):
    W1_ANJLE = "ANKLE"
    W1_ANJLE_num = [0, 0]
    W1_KNEE = "KNEE"
    W1_KNEE_num = [1, 1]
    W1_BUTTOCK = "BUTTOCK"
    W1_BUTTOCK_num = [2, 2]
    W1_WAIST = "WAIST"
    W1_WAIST_num = [3, 3]
    W1_NECK1 = "NECK1"
    W1_NECK2 = "NECK2"
    W1_HEAD_num = [4, 5]
    W1_LEFT_J1 = "LEFT_J1"
    W1_LEFT_J2 = "LEFT_J2"
    W1_LEFT_J3 = "LEFT_J3"
    W1_LEFT_J4 = "LEFT_J4"
    W1_LEFT_J5 = "LEFT_J5"
    W1_LEFT_J6 = "LEFT_J6"
    W1_LEFT_J7 = "LEFT_J7"
    W1_LEFT_num = [6, 12]
    W1_RIGHT_J1 = "RIGHT_J1"
    W1_RIGHT_J2 = "RIGHT_J2"
    W1_RIGHT_J3 = "RIGHT_J3"
    W1_RIGHT_J4 = "RIGHT_J4"
    W1_RIGHT_J5 = "RIGHT_J5"
    W1_RIGHT_J6 = "RIGHT_J6"
    W1_RIGHT_J7 = "RIGHT_J7"
    W1_RIGHT_num = [13, 19]
    



class SharedMemoryKey(Enum):
    """共享内存字典键名枚举"""

    # Kingfisher相关
    KINGFISHER_LEFT = "kingfisher_left"
    KINGFISHER_RIGHT = "kingfisher_right"


class EventKey(Enum):
    """事件名称枚举"""

    READY = "ready_event"
    SHUTDOWN = "shutdown_event"
    OBSERVATIONS_READY = "observations_ready_event"
    ACTIONS_READY = "actions_ready_event"
    START_INFER = "start_infer_event"
    KINGFISHER_READY = "kingfisher_ready_event"
    KINGFISHER_READ = "kingfisher_read_event"
    KINGFISHER_WRITE = "kingfisher_write_event"


class DeployedDexforceVLAKey(Enum):
    """Init Dexforce VLA Model Configurations"""

    MODEL_NAME = "model_name"
    PRETRAINED = "pretrained"
    VISION_ENCODER_PATH = "vision_encoder_path"
    PRECOMPUTE_LANG_EMBEDDINGS = "precompute_lang_embeddings"


class ProcessKey(Enum):
    """进程名称枚举"""

    POLICY_SERVER = "policy_server"
    ROBOT_CLIENT = "robot_client"
    KINGFISHER_PROCESS = "kingfisher_process"


class InferenceConfigKey(Enum):
    """推理配置键名枚举"""

    INFERENCE_LATENCY = "inference_latency"
    ACTION_HORIZON = "action_horizon"
    ROBOT_META = "robot_meta"
    TASK_TYPE = "task_type"
    DEFAULT_QPOS = "default_qpos"
    ARM_ACTION_SHAPE = "shape"
    STATES_DICT_NAMES = "states_dict_names"
    BLENDING_HORIZON = "blending_horizon"
    WAISTQPOS = "waistqpos"
    LEFT_ARMQPOS = "left_armqpos"
    RIGHT_ARMQPOS = "right_armqpos"
    HEADQPOS = "headqpos"
    LEFT_EEFHAND = "left_eefhand"
    RIGHT_EEFHAND = "right_eefhand"
    LEFT_EEFGRIPPER = "left_eefgripper"
    RIGHT_EEFGRIPPER = "right_eefgripper"
    ANKLEQPOS = "ankleqpos"
    BUTTOCKQPOS = "buttockqpos"
    KNEEQPOS = "kneeqpos"
    TORSO = "torso"
    HEAD = "head"
    HAND = "hand"
    GRIPPER = "gripper"

w1qpos_group_map = {
    InferenceConfigKey.LEFT_ARMQPOS.value: ["LEFT_ARM"],
    InferenceConfigKey.RIGHT_ARMQPOS.value: ["RIGHT_ARM"],
    InferenceConfigKey.RIGHT_EEFHAND.value: ["RIGHT_EEFHAND"],
    InferenceConfigKey.LEFT_EEFHAND.value: ["LEFT_EEFHAND"],
    InferenceConfigKey.HEADQPOS.value: ["HEAD"],
    InferenceConfigKey.WAISTQPOS.value: ["WAIST"],
    InferenceConfigKey.RIGHT_EEFGRIPPER.value: ["RIGHT_EEFGRIPPER"],
    InferenceConfigKey.LEFT_EEFGRIPPER.value: ["LEFT_EEFGRIPPER"],
    InferenceConfigKey.ANKLEQPOS.value: ["ANKLE"],
    InferenceConfigKey.KNEEQPOS.value: ["KNEE"],
    InferenceConfigKey.BUTTOCKQPOS.value: ["BUTTOCK"],
}

w1qpos_names_map = {
    # InferenceConfigKey.TORSO.value: ["ANKLE", "KNEE", "BUTTOCK", "WAIST"],
    # InferenceConfigKey.HEAD.value: ["NECK1", "NECK2"],
    InferenceConfigKey.ANKLEQPOS.value: ["ANKLE"],
    InferenceConfigKey.KNEEQPOS.value: ["KNEE"],
    InferenceConfigKey.BUTTOCKQPOS.value: ["BUTTOCK"],
    InferenceConfigKey.WAISTQPOS.value: ["WAIST"],
    InferenceConfigKey.HEADQPOS.value: ["NECK1", "NECK2"],
    InferenceConfigKey.LEFT_ARMQPOS.value: [
        "LEFT_J1",
        "LEFT_J2",
        "LEFT_J3",
        "LEFT_J4",
        "LEFT_J5",
        "LEFT_J6",
        "LEFT_J7",
    ],
    InferenceConfigKey.RIGHT_ARMQPOS.value: [
        "RIGHT_J1",
        "RIGHT_J2",
        "RIGHT_J3",
        "RIGHT_J4",
        "RIGHT_J5",
        "RIGHT_J6",
        "RIGHT_J7",
    ],
    InferenceConfigKey.LEFT_EEFHAND.value: [
        "LEFT_THUMBMCP",
        "LEFT_THUMBCMC",
        "LEFT_INDEXMCP",
        "LEFT_MIDDLEMCP",
        "LEFT_RINGMCP",
        "LEFT_LITTLEMCP",
    ],
    InferenceConfigKey.RIGHT_EEFHAND.value: [
        "RIGHT_THUMBMCP",
        "RIGHT_THUMBCMC",
        "RIGHT_INDEXMCP",
        "RIGHT_MIDDLEMCP",
        "RIGHT_RINGMCP",
        "RIGHT_LITTLEMCP",
    ],
    InferenceConfigKey.LEFT_EEFGRIPPER.value: [
        "FINGER1"
    ],
    InferenceConfigKey.RIGHT_EEFGRIPPER.value: [
        "FINGER1"
    ],
}

camera_used_names_map = {
    ImageKey.CAM_HIGH.value: "head_left_camera",
    ImageKey.CAM_HIGH_R.value: "head_right_camera",
    ImageKey.CAM_HAND_LEFT.value: "left_wrist_camera",
    ImageKey.CAM_HAND_RIGHT.value: "right_wrist_camera",
}



# TODO: read from urdf
JointLimit={
    "WAIST" : [-2.9670597284, 2.9670597284],
    "LEFT_J1" : [-2.9670597284, 2.9670597284],
    "LEFT_J2" : [-2.0943951024, 1.5707963268],
    "LEFT_J3" : [-2.9670597284, 2.9670597284],
    "LEFT_J4" : [-2.3561944902, 1.5707963268],
    "LEFT_J5" : [-2.9670597284, 2.9670597284],
    "LEFT_J6" : [-0.7853981634, 0.7853981634],
    "LEFT_J7" : [-1.5707963268, 1.0471975512],
    "NECK1" : [-1.5707963268, 1.5707963268],
    "NECK2" : [-0.7853981634, 0.4363323130],
    "RIGHT_J1" : [-2.9670597284, 2.9670597284],
    "RIGHT_J2" : [-1.5707963268, 2.0943951024],
    "RIGHT_J3" : [-2.9670597284, 2.9670597284],
    "RIGHT_J4" : [-1.5707963268, 2.3561944902],
    "RIGHT_J5" : [-2.9670597284, 2.9670597284],
    "RIGHT_J6" : [-0.7853981634, 0.7853981634],
    "RIGHT_J7" : [-1.0471975512, 1.5707963268],
    "BUTTOCK" : [-1.9198621772, 1.5707963268],
    "KNEE" : [-2.7052603406, 2.7052603406],
    "ANKLE" : [-1.5707963268, 1.5707963268]
}

class TrajectoryKeys:
    JOINT_KEYS = [
        "ANKLE", "KNEE", "BUTTOCK", "WAIST", "NECK1", "NECK2",
        "LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7",
        "RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7",
    ]

    FINGER_KEYS = [
        "LEFT_HAND_THUMB1", "LEFT_HAND_THUMB2", "LEFT_HAND_INDEX",
        "LEFT_HAND_MIDDLE", "LEFT_HAND_RING", "LEFT_HAND_PINKY",
        "RIGHT_HAND_THUMB1", "RIGHT_HAND_THUMB2", "RIGHT_HAND_INDEX",
        "RIGHT_HAND_MIDDLE", "RIGHT_HAND_RING", "RIGHT_HAND_PINKY",
    ]

    GRIPPER_KEYS = ["LEFT_GRIPPER", "RIGHT_GRIPPER"]

    HAND_JOINT_NAMES = ["T_CMC_YAW", "T_MCP", "IF_MCP_PITCH", "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH"]
    PIPER_LEFT_JOINT = ["PIPER_LEFT"]
    PIPER_RIGHT_JOINT = ["PIPER_RIGHT"]

class ModelNameKeys(Enum):
    VISION_LANGUAGE = "vision_language"
    IMAGES = "images"
    BRAIN = "brain"
    CEREBELLUM = "cerebellum"
    ENCODERS = "encoders"
    