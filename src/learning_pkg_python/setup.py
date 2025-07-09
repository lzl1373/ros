from setuptools import find_packages, setup

package_name = 'learning_pkg_python'

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
    maintainer='sunrise',
    maintainer_email='sunrise@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_pup  = learning_pkg_python.test_pup:main',
            'test_pup_web  = learning_pkg_python.test_pup_web:main',
            'test_sub  = learning_pkg_python.test_sub:main',
            'test_sub_pup  = learning_pkg_python.test_sub_pup:main',
            'test_sub_yolov_web  = learning_pkg_python.test_sub_yolov_web:main',
            'test_sub_yolov_web_http  = learning_pkg_python.test_sub_yolov_web_http:main',
            'test1  = learning_pkg_python.test1:main',
            'test_serial  = learning_pkg_python.test_serial:main',
        ],
    },
)
