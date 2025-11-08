from setuptools import setup

package_name = 'ctrl_tap'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='armaanm',
    maintainer_email='armaanmahajanbg@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'command_vel = ctrl_tap.controls_node_velocity:main',
            'command_acc = ctrl_tap.controls_node_acceleration:main',
            'controls = ctrl_tap.controls_code:main',
            'control_final = ctrl_tap.Control_final:main',
            'control_plotter = ctrl_tap.Control_plotter:main',
            'velocity_profiler = ctrl_tap.controls_velpro:main',
            'path_planner = ctrl_tap.path_points:main',
            'path_view = ctrl_tap.path_view:main',
        ],
    },
)
