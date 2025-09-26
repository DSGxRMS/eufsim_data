from setuptools import setup

package_name = 'gt_tap'

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
            'gt_pose   = gt_tap.gt_pose_node:main',
            'gt_cones  = gt_tap.gt_cones_node:main',
            'gt_wheels = gt_tap.gt_wheels_node:main',
    
        ],
    },
)
