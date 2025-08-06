#include "crow_all.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <iostream>
#include <chrono>
#include "yolov5_detector.hpp"

// 全局变量保存最近一次推理的检测结果图片
cv::Mat latest_result_img;
std::mutex img_mutex;

// 推理线程函数，不断采集、推理并保存最新结果
void detect_worker(BPU_Detect& detector, int cam_id = 0) {
		cv::VideoCapture cap(cam_id); // 采集摄像头（可改为视频流、图片等）
		if (!cap.isOpened()) {
				std::cerr << "[web_server] Failed to open camera " << cam_id << std::endl;
				return;
		}
		while (true) {
				cv::Mat frame, output;
				if (!cap.read(frame)) {
						std::cerr << "[web_server] Failed to read frame from camera" << std::endl;
						break;
				}
				// 推理（绘制结果在 output 中）
				if (detector.Detect(frame, output)) {
						std::lock_guardstd::mutex lock(img_mutex);
						latest_result_img = output.clone();
				}
				// 控制帧率
				std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
				}
}

int main(int argc, char** argv) {
		std::string model_path = "/root/new_ws/src/rdkx5_yolov5/models/detect.bin"; 
		int classes_num = 4; 
		float nms_threshold = 0.45f;
		float score_threshold = 0.25f;
		int nms_top_k = 100;
		int camera_id = 0;
		
		// 初始化检测器
		BPU_Detect detector(model_path, classes_num, nms_threshold, score_threshold, nms_top_k);
		if (!detector.Init()) {
		    std::cerr << "[web_server] Detector init failed!" << std::endl;
		    return -1;
		}
		
		// 启动推理线程
		std::thread worker(detect_worker, std::ref(detector), camera_id);
		
		// Web 服务器
		crow::SimpleApp app;
		
		// 图片接口（返回当前最新检测结果图片）
		CROW_ROUTE(app, "/result.jpg")([]() {
		    std::vector<uchar> buf;
		    {
		        std::lock_guard<std::mutex> lock(img_mutex);
		        if (latest_result_img.empty()) return crow::response(404);
		        cv::imencode(".jpg", latest_result_img, buf);
		    }
		    return crow::response(buf.begin(), buf.end());
		});
		
		// 网页接口
		CROW_ROUTE(app, "/")([]() {
		    return R"(
		        <html>
		        <head><title>YOLOv5 X5 Web可视化</title></head>
		        <body>
		            <h2>YOLOv5 检测结果</h2>
		            <img id='detect' width='640'/>
		            <script>
		            setInterval(() => {
		                document.getElementById('detect').src = '/result.jpg?t=' + Date.now();
		            }, 300);
		            </script>
		        </body>
		        </html>
		    )";
		});
		
		std::cout << "访问 http://<上位机IP>:8080/ 查看检测结果" << std::endl;
		app.port(8080).bindaddr("0.0.0.0").run();
	
	worker.join();
	return 0;
}