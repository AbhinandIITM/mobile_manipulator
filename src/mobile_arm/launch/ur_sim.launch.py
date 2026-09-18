import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5",
            description="Type/series of used UR robot.",
        )
    )

    ur_type = LaunchConfiguration("ur_type")
    
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("mobile_arm"), "urdf", "ur.urdf.xacro"]),
            " ",
            "ur_type:=", ur_type,
            " ",
            "sim_ignition:=true",
            " ",
            "name:=ur",
            " ",
            "simulation_controllers:=",
            PathJoinSubstitution([FindPackageShare("mobile_arm"), "config", "ur_controllers.yaml"]),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # Gazebo Sim
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # Bridge Gazebo clock to ROS 2
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string", robot_description_content,
            "-name", "ur",
            "-allow_renaming", "true",
            "-z", "0.1",
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        parameters=[
            {
                "robot_description": robot_description_content,
                "use_sim_time": True,
            }
        ],
        remappings=[
            (
                "/joint_states",
                "/gui_joint_states"
            )
        ],
        output="screen",
    )

    # Convert GUI JointState messages into JointTrajectory commands
    # for the JointTrajectoryController.

    joint_state_to_trajectory_node = Node(
        package="mobile_arm",
        executable="joint_state_to_trajectory",
        name="joint_state_to_trajectory",
        parameters=[
            {
                "use_sim_time": True,
            }
        ],
        output="screen",
    )
    ik_node = Node(
        package="mobile_arm",
        executable="ik_node",
        name="ik_node",
        parameters=[
            {
                "use_sim_time": True,
            }
        ],
        output="screen",
    )

    nodes = [
        gazebo_launch,
        clock_bridge,
        robot_state_publisher_node,
        spawn_entity,
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner,
        # joint_state_publisher_gui_node,
        # joint_state_to_trajectory_node,
        ik_node,
    ]

    return LaunchDescription(declared_arguments + nodes)