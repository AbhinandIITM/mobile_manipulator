import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except OSError:
        return None

def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type", default="ur5")
    
    # Get URDF via xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([FindPackageShare("mobile_arm"), "urdf", "ur.urdf.xacro"]),
        " ",
        "ur_type:=", ur_type,
        " ",
        "sim_ignition:=true",
        " ",
        "name:=ur"
    ])
    robot_description = {"robot_description": robot_description_content}

    # Get SRDF
    with open(os.path.join(get_package_share_directory("mobile_arm"), "config", "moveit", "ur.srdf"), "r") as f:
        robot_description_semantic_content = f.read()
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    # Get kinematics
    kinematics_yaml = load_yaml("mobile_arm", "config/moveit/kinematics.yaml")
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}

    # Get joint limits
    joint_limits_yaml = load_yaml("mobile_arm", "config/moveit/joint_limits.yaml")
    robot_description_planning = {"robot_description_planning": joint_limits_yaml}

    # Planning Functionality
    planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        }
    }

    # Trajectory Execution Functionality
    controllers_yaml = load_yaml("mobile_arm", "config/moveit/moveit_controllers.yaml")
    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_yaml["moveit_simple_controller_manager"],
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }
    
    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            {"use_sim_time": True},
        ],
    )

    # RViz
    rviz_config_file = PathJoinSubstitution([FindPackageShare("mobile_arm"), "config", "moveit", "moveit.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_planning,
            robot_description_kinematics,
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([move_group_node, rviz_node])
