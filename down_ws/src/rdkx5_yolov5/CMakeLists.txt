cmake_minimum_required(VERSION 3.8)
project(rdkx5_yolov5)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置编译类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# 设置编译选项
set(CMAKE_CXX_FLAGS_DEBUG " -Wall -g -O0 ")
set(CMAKE_C_FLAGS_DEBUG " -Wall -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE " -Wall -O3 ")
set(CMAKE_C_FLAGS_RELEASE " -Wall -O3 ")
# libdnn.so 依赖设置
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vision_msgs REQUIRED)

set(dependencies
    rclcpp
    std_msgs
    sensor_msgs
    cv_bridge
    image_transport
    OpenCV
    vision_msgs
)

# 设置RDK BPU库路径
set(DNN_PATH "/usr/include/dnn")      # BPU头文件路径
set(DNN_LIB_PATH "/usr/lib/")         # BPU库文件路径
# 添加头文件路径
include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rdkx5_yolov5
    ${DNN_PATH}
)
# 添加源文件
add_executable(yolov5_node 
    src/yolov5_node.cpp
    src/yolov5_detector.cpp
)

add_executable(yolov5_web_server 
		src/web_server.cpp 
		src/yolov5_detector.cpp
)

ament_target_dependencies(yolov5_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
  vision_msgs
)

target_include_directories(yolov5_web_server PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rdkx5_yolov5
)

# 链接依赖库
target_link_libraries(yolov5_node
    dnn                # RDK BPU库
    pthread            # 线程库
    rt                 # 实时库
    dl                 # 动态链接库
)
target_link_libraries(yolov5_web_server 
		${OpenCV_LIBS} 
		dnn                
    pthread          
    rt                
    dl                 
)
# 安装目标
install(TARGETS
  yolov5_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    launch
  DESTINATION
    share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()