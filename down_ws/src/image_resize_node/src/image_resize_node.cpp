// Copyright (c) 2024，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <vector>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <hobot_cv/hobotcv_imgproc.h>

class ImageResizeNode : public rclcpp::Node
{
public:
  ImageResizeNode() : Node("image_resize_node")
  {
    RCLCPP_INFO(this->get_logger(), "Image resize node starting...");
    
    // 配置订阅者的 QoS 策略
    rclcpp::QoS qos_profile(10);  // 历史深度为10
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
    // qos_profile.lifespan(rclcpp::Duration::max());  // Infinite lifespan
    // qos_profile.deadline(rclcpp::Duration::max());  // Infinite deadline
    qos_profile.liveliness(RMW_QOS_POLICY_LIVELINESS_AUTOMATIC);
    // qos_profile.liveliness_lease_duration(rclcpp::Duration::max());  // Infinite lease duration
    
    // 创建订阅者 - 订阅HbmMsg1080P消息
    subscription_ = this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
      "/hbmem_img", qos_profile,
      std::bind(&ImageResizeNode::image_callback, this, std::placeholders::_1));
    
    // 创建发布者 - 发布CompressedImage消息
    publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
      "/image_resized/compressed", 10);
    
    RCLCPP_INFO(this->get_logger(), "Image resize node initialized successfully");
    RCLCPP_INFO(this->get_logger(), "Subscribing to: /hbmem_img");
    RCLCPP_INFO(this->get_logger(), "Publishing to: /image_resized/compressed");
  }

private:
  void image_callback(hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg)
  {
    RCLCPP_DEBUG(this->get_logger(), "Image callback triggered");
    
    try {
      // 记录开始时间
      auto start_time = std::chrono::high_resolution_clock::now();
      
      // 检查输入图像数据是否有效
      if (msg->data.empty() || msg->height == 0 || msg->width == 0) {
        RCLCPP_WARN(this->get_logger(), "Invalid input image data");
        return;
      }
      
      RCLCPP_DEBUG(this->get_logger(), "Received image: %dx%d, data_size: %d", 
                   msg->width, msg->height, msg->data_size);
      
      // 将 NV12 数据转换为 cv::Mat
      // NV12 格式: Y 分量在前, UV 交错在后
      cv::Mat nv12_mat(msg->height*3/2, msg->width, CV_8UC1, msg->data.data());
      
      
      // 使用 hobot_cv 的 resize 函数进行硬件加速 resize
      cv::Mat resized_nv12;
      hobot_cv::hobotcv_resize(
        nv12_mat,      
        msg->height,           
        msg->width,            
        resized_nv12,    
        msg->height/4,              
        msg->width/4             
      );
      
      // 检查 resize 是否成功
      if (resized_nv12.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to resize image");
        return;
      }
      
      // 编码为JPEG格式
      std::vector<uchar> compressed_data;
      std::vector<int> compression_params;

      // 转换 NV12 到 BGR
      cv::Mat resized_bgr;
      cv::cvtColor(resized_nv12, resized_bgr, cv::COLOR_YUV2BGR_NV12);

      compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
      compression_params.push_back(60);  // JPEG质量设置为90
      
      bool encode_success = cv::imencode(".jpg", resized_bgr, compressed_data, compression_params);
      
      if (!encode_success) {
        RCLCPP_ERROR(this->get_logger(), "Failed to encode image to JPEG");
        return;
      }
      
      // 创建CompressedImage消息
      auto compressed_msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
      compressed_msg->header.stamp = msg->time_stamp;
      compressed_msg->header.frame_id = "default_usb_cam";
      compressed_msg->format = "jpeg";
      compressed_msg->data = compressed_data;
      
      // 发布消息
      publisher_->publish(*compressed_msg);
      
      // 计算处理时间
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      
      RCLCPP_DEBUG(this->get_logger(), 
                   "Successfully processed and published resized image: %dx%d -> 640x480, processing time: %ldms",
                   msg->width, msg->height, duration.count());
      
      // 只在第一次处理时输出信息级别的日志
      static bool first_process = true;
      if (first_process) {
        RCLCPP_INFO(this->get_logger(), 
                    "First image processed successfully: %dx%d -> 640x480, processing time: %ldms",
                    msg->width, msg->height, duration.count());
        first_process = false;
      }
    }
    catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Exception in image processing: %s", e.what());
    }
    catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Unknown exception in image processing");
    }
  }

  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  
  RCLCPP_INFO(rclcpp::get_logger("image_resize_node"), "Starting Image Resize Node...");
  
  auto node = std::make_shared<ImageResizeNode>();
  
  RCLCPP_INFO(rclcpp::get_logger("image_resize_node"), "Image Resize Node started successfully");
  
  try {
    rclcpp::spin(node);
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("image_resize_node"), "Exception during spin: %s", e.what());
  }
  
  rclcpp::shutdown();
  RCLCPP_INFO(rclcpp::get_logger("image_resize_node"), "Image Resize Node shutdown");
  return 0;
}