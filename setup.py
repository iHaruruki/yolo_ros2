import os
from glob import glob
from pathlib import Path
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
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
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
            'image_show_node = yolo_ros2.imshow:main',
            'object_detection_node = yolo_ros2.object_detection:main',
            'object_segmentation_node = yolo_ros2.object_segmentation:main',
            'object_detection_tf_node = yolo_ros2.object_detection_tf:main',
        ],
    },
)
