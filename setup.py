from setuptools import find_packages, setup

package_name = 'yolo_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/yolo_depth.launch.py']),
    ],
    install_requires=[
           'setuptools',
           'ultralytics',
           'opencv-python',
    ],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='ryo.saegusa@syblab.org',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_node = yolo_ros2.yolo_node:main',
            'yolo_depth_node = yolo_ros2.yolo_depth_node:main',
        ],
    },
)
