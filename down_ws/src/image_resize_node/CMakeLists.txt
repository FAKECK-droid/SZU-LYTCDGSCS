cmake_minimum_required(VERSION 3.8)
project(image_resize_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(hbm_img_msgs REQUIRED)
find_package(PkgConfig REQUIRED)

# Find OpenCV using pkg-config
pkg_check_modules(OpenCV REQUIRED opencv4)

# Set library paths manually for TROS system
set(TROS_LIB_DIR "/opt/tros/humble/lib")
set(HOBOT_CV_LIB "${TROS_LIB_DIR}/libhobot_cv.so")
set(DNN_LIB "${TROS_LIB_DIR}/libdnn_node.so")

# Check if libraries exist
if(NOT EXISTS ${HOBOT_CV_LIB})
  message(FATAL_ERROR "Could not find hobot_cv library at ${HOBOT_CV_LIB}")
endif()
if(NOT EXISTS ${DNN_LIB})
  message(FATAL_ERROR "Could not find dnn library at ${DNN_LIB}")
endif()

# Add executable
add_executable(image_resize_node src/image_resize_node.cpp)

# Include directories
target_include_directories(image_resize_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  /opt/tros/humble/include
)

# Link libraries
target_link_libraries(image_resize_node
  ${OpenCV_LIBRARIES}
  ${HOBOT_CV_LIB}
  ${DNN_LIB}
)

# Specify dependencies
ament_target_dependencies(image_resize_node
  rclcpp
  sensor_msgs
  hbm_img_msgs
)

# Install targets
install(TARGETS image_resize_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
