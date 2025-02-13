from setuptools import setup

package_name = 'brt_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[],
    zip_safe=True,
    maintainer='krri',
    maintainer_email='NONE',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'sit_recognition = brt_manager.sit_recognition:main',
            'board = brt_manager.board:main',
            'data = brt_manager.Data_collecting_origin:main',
            'test = brt_manager.Data_collecting_thread:main',
            'sync = brt_manager.ApproximateTimeSync:main',
        ],
    },
)
