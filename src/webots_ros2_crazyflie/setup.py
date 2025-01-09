"""webots_ros2 package setup file."""

from setuptools import setup
import os

package_name = 'webots_ros2_crazyflie'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/robot_launch.py']))
data_files.append(('share/' + package_name + '/worlds', [
    'worlds/crazyflie.wbt', 'worlds/.crazyflie.wbproj',
]))
data_files.append(('share/' + package_name + '/resource', [
    'resource/webots_ros2_crazyflie.urdf'
]))
data_files.append(('share/' + package_name, ['package.xml']))

# Rekurencyjne dodanie plików z folderu `protos`
for root, dirs, files in os.walk('/protos'):
    install_path = os.path.join('/share', package_name, root)  # Ścieżka docelowa w pakiecie
    if files or dirs:  # Dodaj tylko, jeśli są pliki
        file_paths = [os.path.join(root, f) for f in files]
        data_files.append((install_path, file_paths))

setup(
    name=package_name,
    version='2023.1.1',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools', 'launch'],
    zip_safe=True,
    author='Matthew Lock',
    author_email='mlwock@kth.se',
    maintainer='Matthew Lock',
    maintainer_email='mlwock@kth.se',
    keywords=['ROS', 'Webots', 'Robot', 'Simulation', 'Examples'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Bitcraze Crazyflie ROS2 interface for Webots.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'launch.frontend.launch_extension': ['launch_ros = launch_ros']
    }
)
