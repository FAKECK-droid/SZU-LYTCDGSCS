#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include "rdkx5_yolov5/yolov5_detector.hpp"

class YOLOv5Node : public rclcpp::Node {
public:
    YOLOv5Node() : Node("yolov5_node") {
        // 参数声明
        this->declare_parameter("model_path", "/root/new_ws/src/rdkx5_yolov5/models/detect.bin");
        this->declare_parameter("classes_num", 4);
        this->declare_parameter("nms_threshold", 0.45f);
        this->declare_parameter("score_threshold", 0.25f);
        this->declare_parameter("nms_top_k", 300);

        // 获取参数
        std::string model_path = this->get_parameter("model_path").as_string();
        int classes_num = this->get_parameter("classes_num").as_int();
        float nms_threshold = this->get_parameter("nms_threshold").as_double();
        float score_threshold = this->get_parameter("score_threshold").as_double();
        int nms_top_k = this->get_parameter("nms_top_k").as_int();   

        // 初始化检测器
        detector_ = std::make_unique<BPU_Detect>(
            model_path, classes_num, nms_threshold, score_threshold, nms_top_k);

        if (!detector_->Init()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize BPU detector");
            rclcpp::shutdown();
            return;
        }

        // 创建订阅者
        image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/image", 10, std::bind(&YOLOv5Node::imageCallback, this, std::placeholders::_1));

        // 创建发布者
        detections_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "detections", 10);
        result_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "result_image", 10);

        RCLCPP_INFO(this->get_logger(), "YOLOv5 node initialized successfully");
    }

private:
    void imageCallback(
        const sensor_msgs::msg::CompressedImage::ConstSharedPtr& img_msg) {

        try {
            // 转换ROS图像消息为OpenCV格式
            cv::Mat frame = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
            cv::Mat output_frame;

            // 执行目标检测
            if (!detector_->Detect(frame, output_frame)) {
                RCLCPP_ERROR(this->get_logger(), "Detection failed");
                return;
            }

            // 发布检测结果
            publishDetections(img_msg->header);

            // 发布结果图像
            auto result_img_msg = cv_bridge::CvImage(
                img_msg->header, "bgr8", output_frame).toImageMsg();
            result_img_pub_->publish(*result_img_msg);

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge error: %s", e.what());
        }
    }

    void publishDetections(const std_msgs::msg::Header& header) {
        vision_msgs::msg::Detection2DArray detections_msg;
        detections_msg.header = header;

        // 获取检测器的检测结果
        const auto& bboxes = detector_->GetDetections();
        const auto& scores = detector_->GetScores();
        const auto& indices = detector_->GetIndices();
        const auto& class_names = detector_->GetClassNames();

        // 获取缩放和偏移参数
        const float x_scale = detector_->GetXScale();
        const float y_scale = detector_->GetYScale();
        const int x_shift = detector_->GetXShift();
        const int y_shift = detector_->GetYShift();

        // 转换检测结果为ROS消息
        for (int cls_id = 0; cls_id < detector_->GetClassesNum(); cls_id++) {
            for (size_t i = 0; i < indices[cls_id].size(); i++) {
                int idx = indices[cls_id][i];

                // 计算原始图像坐标
                float x1 = (bboxes[cls_id][idx].x - x_shift) / x_scale;
                float y1 = (bboxes[cls_id][idx].y - y_shift) / y_scale;
                float width = bboxes[cls_id][idx].width / x_scale;
                float height = bboxes[cls_id][idx].height / y_scale;

                // 创建Detection2D消息
                vision_msgs::msg::Detection2D detection;
                detection.header = header;

                // 设置边界框
                detection.bbox.center.position.x = x1 + width / 2;
                detection.bbox.center.position.y = y1 + height / 2;
                detection.bbox.size_x = width;
                detection.bbox.size_y = height;

                // 设置类别和置信度
                vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
                hypothesis.hypothesis.class_id = class_names[cls_id];
                hypothesis.hypothesis.score = scores[cls_id][idx];
                detection.results.push_back(hypothesis);

                detections_msg.detections.push_back(detection);
            }
        }

        // 发布检测结果
        detections_pub_->publish(detections_msg);
    }

    // ROS 2订阅者和发布者
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;

    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr result_img_pub_;

    // 检测器实例
    std::unique_ptr<BPU_Detect> detector_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YOLOv5Node>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}