#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ament_index_python.packages import get_package_share_directory
import ikpy.chain

class IKNode(Node):
    def __init__(self):
        super().__init__('ik_node')
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]
        
        # Load URDF via xacro
        self.get_logger().info('Generating URDF for IK solver...')
        pkg_share = get_package_share_directory('mobile_arm')
        xacro_file = os.path.join(pkg_share, 'urdf', 'ur.urdf.xacro')
        urdf_str = os.popen(f'xacro {xacro_file} name:=ur ur_type:=ur5 sim_ignition:=true').read()
        urdf_path = '/tmp/mobile_arm_ik.urdf'
        with open(urdf_path, 'w') as f:
            f.write(urdf_str)
            
        # The UR5 active links mask
        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path, 
            active_links_mask=[False, False, True, True, True, True, True, True, False]
        )
        self.get_logger().info('IK chain initialized with joints: ' + str([l.name for l in self.chain.links]))
        
        self.subscription = self.create_subscription(
            Pose,
            '/target_pose',
            self.pose_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Initial joint positions
        self.current_joints = [0.0] * len(self.chain.links)

        self.get_logger().info('IK Node started. Send Pose messages to /target_pose to move the arm.')

    def pose_callback(self, msg: Pose):
        target_position = [msg.position.x, msg.position.y, msg.position.z]
        
        # We calculate the inverse kinematics for the target position
        ik_solution = self.chain.inverse_kinematics(
            target_position=target_position,
            initial_position=self.current_joints
        )
        self.current_joints = ik_solution
        
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.header.stamp = self.get_clock().now().to_msg()
        
        point = JointTrajectoryPoint()
        
        # Extract the active joint positions
        active_positions = [ik_solution[i] for i in range(len(ik_solution)) if self.chain.links[i].name in self.joint_names]
        
        point.positions = active_positions
        # A duration of 1 second for the motion
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0
        
        trajectory.points = [point]
        self.publisher.publish(trajectory)
        self.get_logger().info(f'Published IK solution for target [{target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f}]')

def main(args=None):
    rclpy.init(args=args)
    node = IKNode()
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
