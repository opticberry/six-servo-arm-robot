from setuptools import find_packages, setup

package_name = 'six_servo_arm_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='orangepi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		'ppo_local_servo_node = six_servo_arm_rl.ppo_local_servo_node:main',
        'ppo_local_servo_node2 = six_servo_arm_rl.ppo_local_servo_node2:main',
        ],
    },
)
