from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    clearpath_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('clearpath_gz'),
                'launch',
                'simulation.launch.py'
            ])
        ]),
        launch_arguments={
            'setup_path': PathJoinSubstitution([FindPackageShare('j100_sim'), 'config']),
            'world': 'empty'
            # 'world': 'warehouse'
        }.items()
    )

    return LaunchDescription([
        clearpath_sim
    ])
