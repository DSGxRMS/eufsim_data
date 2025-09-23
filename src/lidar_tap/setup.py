from setuptools import setup

package_name = 'lidar_tap'

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
            'velodyne_tap = lidar_tap.pointcloud_velodyne:main',
            'pcviewer = lidar_tap.lidarview:main',
            'lidardata = lidar_tap.lidardata:main'
        ],
    },
)
