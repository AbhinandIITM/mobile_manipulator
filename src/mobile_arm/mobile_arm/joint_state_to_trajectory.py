#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class JointStateToTrajectory(Node):

    def __init__(self):
        super().__init__('joint_state_to_trajectory')

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

        self.positions = {
            joint: 0.0
            for joint in self.joint_names
        }

        self.received_joints = set()

        self.subscription = self.create_subscription(
            JointState,
            '/gui_joint_states',
            self.joint_state_callback,
            10
        )

        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.get_logger().info(
            'JointState -> JointTrajectory relay started'
        )
        self.get_logger().info(
            'Input : /gui_joint_states'
        )
        self.get_logger().info(
            'Output: /joint_trajectory_controller/joint_trajectory'
        )

    def joint_state_callback(self, msg: JointState):

        for i, name in enumerate(msg.name):

            if name not in self.positions:
                continue

            if i >= len(msg.position):
                continue

            self.positions[name] = msg.position[i]
            self.received_joints.add(name)

        # Wait until the GUI has supplied all six joints.
        if not all(
            joint in self.received_joints
            for joint in self.joint_names
        ):
            return

        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)
        
        # Add the current time stamp to the header to prevent the controller from dropping the message
        trajectory.header.stamp = self.get_clock().now().to_msg()

        point = JointTrajectoryPoint()

        point.positions = [
            self.positions[joint]
            for joint in self.joint_names
        ]

        # 250 ms trajectory.
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 250_000_000

        trajectory.points = [point]

        self.publisher.publish(trajectory)


def main(args=None):

    rclpy.init(args=args)

    node = JointStateToTrajectory()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()