第二十届全国大学生智能车竞赛地平线智慧医疗赛道深圳大学 荔园天才地瓜赛车手队伍源代码

down_ws文件夹为下位机小车工作空间源代码 up文件夹内为上位机程序（使用需开启wsl桥接模式）

#启动摄像头、推理节点、底盘、rosbridge、二维码 ros2 launch race_states_holder race_holder.launch.py

#启动运动控制 ros2 run race_states_holder motion_control_pid

#启动图像压缩节点 ros2 run image_resize_node image_resize_node
