import rclpy
from rclpy.node import Node
from joint_interfaces.msg import JointPositionControl
from std_msgs.msg import Float64MultiArray

joint_keys = [
    "ANKLE", "KNEE", "BUTTOCK", "WAIST", "NECK1", "NECK2", "LEFT_J1", "LEFT_J2",
    "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7", "RIGHT_J1",
    "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7"
]


class JointCommandPublisher:

    def __init__(self, node_name="replay_node"):
        rclpy.init(args=None)
        self.node = rclpy.create_node(node_name)
        self.publisher = self.node.create_publisher(JointPositionControl,
                                                    "/control/joint_position",
                                                    10)
        self.left_publisher = self.node.create_publisher(
            Float64MultiArray, '/set_brainco_left_hand_qpos', 10)
        self.right_publisher = self.node.create_publisher(
            Float64MultiArray, '/set_brainco_right_hand_qpos', 10)
        self.VR_head_publisher = self.node.create_publisher(
            Float64MultiArray, "/teleop/VR_head", 10)
        self.VR_left_hand_publisher = self.node.create_publisher(
            Float64MultiArray, "/teleop/VR_left_hand", 10)
        self.VR_right_hand_publisher = self.node.create_publisher(
            Float64MultiArray, "/teleop/VR_right_hand", 10)
        self.left_gripper_publisher = self.node.create_publisher(
            Float64MultiArray, '/set_left_pgc_position', 10)
        self.right_gripper_publisher = self.node.create_publisher(
            Float64MultiArray, '/set_right_pgc_position', 10)
        self.joint_keys = joint_keys

    def send_joint_position(self, joint_data: dict, frame_id):
        msg = JointPositionControl()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = str(frame_id)
        for key in self.joint_keys:
            if key in joint_data:  # and key not in ['ANKLE', 'KNEE', 'BUTTOCK', 'WAIST', 'NECK1', 'NECK2']:
                msg.name.append(key)
                msg.position.append(joint_data[key])
        # self.node.get_logger().info(f"Publishing joint command: {list(zip(msg.name, msg.position))}")
        self.publisher.publish(msg)

    def send_left_hand_angles(self, angles):
        msg = Float64MultiArray()
        msg.data = angles
        self.left_publisher.publish(msg)

    def send_right_hand_angles(self, angles):
        msg = Float64MultiArray()
        msg.data = angles
        self.right_publisher.publish(msg)

    def send_left_gripper(self, val: float):
        msg = Float64MultiArray()
        msg.data = [val]
        self.left_gripper_publisher.publish(msg)

    def send_right_gripper(self, val: float):
        msg = Float64MultiArray()
        msg.data = [val]
        self.right_gripper_publisher.publish(msg)

    def send_VR_head(self, head_data: list):
        msg = Float64MultiArray()
        msg.data = head_data
        self.VR_head_publisher.publish(msg)

    def send_VR_left_hand(self, left_hand_data: list):
        msg = Float64MultiArray()
        msg.data = left_hand_data
        self.VR_left_hand_publisher.publish(msg)

    def send_VR_right_hand(self, right_hand_data: list):
        msg = Float64MultiArray()
        msg.data = right_hand_data
        self.VR_right_hand_publisher.publish(msg)

    def spin_once(self):
        rclpy.spin_once(self.node, timeout_sec=0.1)

    def destroy(self):
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    import time
    pub = JointCommandPublisher()

    try:
        value = 2.7
        while value <= 3.2:
            joint_data = {"NECK2": value}
            pub.send_joint_position(joint_data)
            # pub.spin_once()
            time.sleep(1.0)  # 1000ms
            value += 0.1
            value = round(value, 1)  # 避免浮点精度误差
    finally:
        pub.destroy()
