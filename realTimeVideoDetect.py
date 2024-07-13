import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import time
import mss
from screen_show import OverlayText 
import config  

# 定义类别名称和对应颜色的映射关系
class_names = {
    0: "GoldFlash",
    1: "RedFlash",
    2: "NoFlash"
}

class_colors = {
    0: (255, 165, 0),  # GoldFlash - 橙色
    1: (255, 0, 0),    # RedFlash - 红色
    2: (0, 0, 0)       # NoFlash - 黑色
}

# 定义预处理函数
def preprocess_image(image, target_size=(640, 360)):
    # 确保图像是RGB图像
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 缩放图像
    image = cv2.resize(image, target_size)
    
    # 转换为 PIL 图像
    image = Image.fromarray(image)
    
    # 定义验证时的转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 应用转换
    image = transform(image).unsqueeze(0)  # 增加批次维度
    return image.numpy()

# 初始化 ONNX 模型
def initialize_model(model_path):
    ort_session = ort.InferenceSession(model_path)
    return ort_session

# 定义实时视频处理函数
def process_video(model_path):
    # 初始化模型
    ort_session = initialize_model(model_path)
    
    # 初始化文本覆盖层
    overlay = OverlayText()

    # 打开屏幕捕获
    with mss.mss() as sct:
        # 获取屏幕尺寸
        monitor = sct.monitors[1]

        # 主循环
        while True:
            start_time = time.time()

            # 捕获屏幕
            screen = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

            # 预处理帧
            input_tensor = preprocess_image(frame)

            # 获取预测结果
            preds = get_prediction(ort_session, input_tensor)

            # 获取分类结果 (假设 preds 是一个列表，取第一个元素作为分类结果)
            class_result = np.argmax(preds[0])

            # 获取类别名称和颜色
            class_name = class_names.get(class_result, "Unknown")
            text_color = class_colors.get(class_result, (255, 255, 255))  # 默认为白色

            # 计算帧率
            end_time = time.time()
            fps = 1 / (end_time - start_time)

            # 更新显示文本，并设置颜色
            overlay.update_text(f'{class_name}, FPS: {fps:.2f}', text_color)




# 获取模型预测结果
def get_prediction(ort_session, input_tensor):
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

# 指定模型路径并运行视频处理
model_path = config.ONXX_MODEL_PATH
process_video(model_path)
