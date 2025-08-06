from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time
import rclpy
from rclpy.node import Node
from qwen_vl_msgs.srv import QwenVL
import cv_bridge
import cv2

class QwenVLNode(Node):
    def __init__(self):
        super().__init__("qwen_vl_server")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "ChineseAlpacaGroup/Qwen2.5-VL-3B-Instruct-GPTQ-Int4",
            device_map="auto",
        )

        self.min_pixels = 128*28*28
        self.max_pixels = 256*28*28
        self.processor = AutoProcessor.from_pretrained("/home/rdk/.cache/modelscope/hub/models/ChineseAlpacaGroup/Qwen2___5-VL-3B-Instruct-GPTQ-Int4", 
                                                    min_pixels=self.min_pixels, 
                                                    max_pixels=self.max_pixels,
                                                    use_fast=True)
        self.bridge = cv_bridge.CvBridge()

        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "image.png",
                    },
                    {"type": "text", "text": "描述这张图片中病人的表情、穿着、动作、状态。输出结构化且简洁高效,50字以内"},
                ],
            }
        ]

        self.server = self.create_service(QwenVL,"/qwen_vl_service",self.qwen_service_callback)
        self.get_logger().info("大模型服务端已启动")

    def qwen_service_callback(self,request,response):
        self.get_logger().info(f"收到请求，开始推理")
        img_msg = request.image
        img_cv2 = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        cv2.imwrite("image_predict.png",img_cv2)

        self.messages[0]["content"][0]["image"] = "image_predict.png"

        #预处理
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(self.messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        #推理加后处理
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response.text = output_text[0]
        print(output_text)
        return response 

if __name__ == "__main__":
    rclpy.init()
    node = QwenVLNode()
    rclpy.spin(node)
    rclpy.shutdown()
