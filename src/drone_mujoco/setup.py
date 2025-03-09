from setuptools import find_packages, setup

package_name = 'drone_mujoco'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kuba',
    maintainer_email='jakub.junkiert@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim = drone_mujoco.sim:main',
            'teleop = drone_mujoco.drone_teleop:main',
            'tuning = drone_mujoco.PID_Tuner:main'
        ],
    },
)
