import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import time
import mss
from screen_show import OverlayText
from model import get_model 
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
    return image

# 初始化 PyTorch 模型
def initialize_model(model_path, device):
    input_height = config.INPUT_SIZE_HIGH
    input_width = config.INPUT_SIZE
    num_classes = 3

    model = get_model(model_name='conformer', input_height=input_height, input_width=input_width, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 执行键盘输入动作
def execute_keyboard_input(key_code):
    win32api.keybd_event(key_code, 0, 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)

# 定义实时视频处理函数
def process_video(model_path):
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = initialize_model(model_path, device)
    
    # 初始化文本覆盖层
    overlay = OverlayText()

    # 初始化计数器
    frame_count = 0
    consecutive_prediction_count = 0
    last_prediction = None

    # 打开屏幕捕获
    with mss.mss() as sct:
        # 获取屏幕尺寸
        monitor = sct.monitors[1]

        # 主循环
        while frame_count < config.MAX_FRAME:
            start_time = time.time()

            # 捕获屏幕
            screen = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

            # 预处理帧
            input_tensor = preprocess_image(frame).to(device)

            # 获取预测结果
            with torch.no_grad():
                preds = model(input_tensor)
                class_result = preds.argmax(dim=1).item()

            # 获取类别名称和颜色
            class_name = class_names.get(class_result, "Unknown")
            text_color = class_colors.get(class_result, (255, 255, 255))  # 默认为白色

            # 检查连续相同预测
            if class_result == last_prediction:
                consecutive_prediction_count += 1
            else:
                consecutive_prediction_count = 1
                last_prediction = class_result

            # 当连续预测达到阈值时执行键盘输入
            if consecutive_prediction_count >= config.SAME_PREDICTION_THRESHOLD:
                if class_name == "RedFlash":
                    execute_keyboard_input(config.RED_INPUT)
                elif class_name == "GoldFlash":
                    execute_keyboard_input(config.GOLD_INPUT)
                consecutive_prediction_count = 0  # 重置计数器

            # 计算帧率
            end_time = time.time()
            fps = 1 / (end_time - start_time)

            # 更新显示文本，并设置颜色
            overlay.update_text(f'{class_name}, FPS: {fps:.2f}', text_color)

            frame_count += 1

# 指定模型路径并运行视频处理
model_path = config.PTH_MODEL_PATH
process_video(model_path)
