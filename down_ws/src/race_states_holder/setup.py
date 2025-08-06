from setuptools import find_packages, setup

package_name = 'race_states_holder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/race_holder.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rdk',
    maintainer_email='rdk@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_control_pid=race_states_holder.motion_control_pid:main',
            'states_holder=race_states_holder.states_holder:main',
        ],
    },
)
