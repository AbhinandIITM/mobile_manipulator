import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mobile_arm'

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

def package_files(data_files, directory_list):
    paths_dict = {}
    for directory in directory_list:
        for (path, directories, filenames) in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)
                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)
                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=package_files(data_files, ['urdf/', 'meshes/', 'config/', 'launch/']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abhi',
    maintainer_email='abhinandt2017@gmail.com',
    description='mobile_arm description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'joint_state_to_trajectory = mobile_arm.joint_state_to_trajectory:main',
        'ik_node = mobile_arm.ik_node:main',
    ],
},
)
