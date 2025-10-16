from setuptools import setup

package_name = 'cam_tap'

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
            'left_cam = cam_tap.left_feed_node:main',
            'right_cam = cam_tap.right_feed_node:main',
            'cam_record = cam_tap.feed_record:main',
            'bbox = cam_tap.bbox_data:main',
        ],
    },
)
