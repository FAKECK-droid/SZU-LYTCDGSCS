#ifndef YOLOV5_DETECTOR_HPP
#define YOLOV5_DETECTOR_HPP
#define ENABLE_DRAW 1

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"
#include <memory>
#include <utility>
#include "vision_msgs/msg/detection2_d_array.hpp"

// 模型和检测相关的默认参数定义
#define DEFAULT_MODEL_PATH "/root/new_ws/src/rdkx5_yolov5/models/detect.bin"  // 默认模型路径
#define DEFAULT_CLASSES_NUM 4          // 默认类别数量
#define CLASSES_LIST "line, roadblock, qrcode, end"     // 类别名称
#define DEFAULT_NMS_THRESHOLD 0.45f    // 非极大值抑制阈值
#define DEFAULT_SCORE_THRESHOLD 0.25f  // 置信度阈值
#define DEFAULT_NMS_TOP_K 300          // NMS保留的最大框数
#define DEFAULT_FONT_SIZE 1.0f         // 绘制文字大小
#define DEFAULT_FONT_THICKNESS 1.0f    // 绘制文字粗细
#define DEFAULT_LINE_SIZE 2.0f         // 绘制线条粗细

// 错误检查宏
#define RDK_CHECK_SUCCESS(value, errmsg) \
    do { \
        auto ret_code = value; \
        if (ret_code != 0) { \
            std::cout << errmsg << ", error code:" << ret_code; \
            return ret_code; \
        } \
    } while (0);

class BPU_Detect {
public:
    // 构造函数
    BPU_Detect(const std::string& model_path = DEFAULT_MODEL_PATH,
               int classes_num = DEFAULT_CLASSES_NUM,
               float nms_threshold = DEFAULT_NMS_THRESHOLD,
               float score_threshold = DEFAULT_SCORE_THRESHOLD,
               int nms_top_k = DEFAULT_NMS_TOP_K);
    
    // 析构函数
    ~BPU_Detect();

    // 获取检测结果的接口方法
    const std::vector<std::vector<cv::Rect2d>>& GetDetections() const { return bboxes_; }
    const std::vector<std::vector<float>>& GetScores() const { return scores_; }
    const std::vector<std::vector<int>>& GetIndices() const { return indices_; }
    const std::vector<std::string>& GetClassNames() const { return class_names_; }
    int GetClassesNum() const { return classes_num_; }
    float GetXScale() const { return x_scale_; }
    float GetYScale() const { return y_scale_; }
    int GetXShift() const { return x_shift_; }
    int GetYShift() const { return y_shift_; }    

    // 主要接口
    bool Init();
    bool Detect(const cv::Mat& input_img, cv::Mat& output_img);
    bool Release();

private:
    // 内部方法
    bool LoadModel();
    bool GetModelInfo();
    bool PreProcess(const cv::Mat& input_img);
    bool Inference();
    bool PostProcess();
    void ReleaseCurrentFrame();
    void DrawResults(cv::Mat& img);
    void PrintResults() const;
    void ProcessFeatureMap(hbDNNTensor& output_tensor, 
                         int height, int width,
                         const std::vector<std::pair<double, double>>& anchors,
                         float conf_thres_raw);

    // 成员变量
    std::string model_path_;
    int classes_num_;
    float nms_threshold_;
    float score_threshold_;
    int nms_top_k_;
    bool is_initialized_;
    float font_size_;
    float font_thickness_;
    float line_size_;
    
    // BPU相关
    hbPackedDNNHandle_t packed_dnn_handle_;
    hbDNNHandle_t dnn_handle_;
    const char* model_name_;
    
    // 输入输出
    hbDNNTensor input_tensor_;
    hbDNNTensor* output_tensors_;
    hbDNNTensorProperties input_properties_;
    
    // 任务
    hbDNNTaskHandle_t task_handle_;
    
    // 模型参数
    int input_h_;
    int input_w_;
    
    // 检测结果
    std::vector<std::vector<cv::Rect2d>> bboxes_;
    std::vector<std::vector<float>> scores_;
    std::vector<std::vector<int>> indices_;
    
    // 图像处理
    float x_scale_;
    float y_scale_;
    int x_shift_;
    int y_shift_;
    cv::Mat resized_img_;
    
    // Anchors
    std::vector<std::pair<double, double>> s_anchors_;
    std::vector<std::pair<double, double>> m_anchors_;
    std::vector<std::pair<double, double>> l_anchors_;
    
    // 输出顺序
    int output_order_[3];
    std::vector<std::string> class_names_;
};

#endif // YOLOV5_DETECTOR_HPP