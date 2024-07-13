import os
import torch
import cv2
import numpy as np
from model import get_model  
import config  
from PIL import Image
import torchvision.transforms as transforms
import time

# 定义加载图片的函数
def load_image_paths(directory):
    """加载指定目录中的所有图片路径"""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# 定义预处理函数
def preprocess_image(image, target_size=(640,360)):
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
    image = transform(image)
    
    # 增加 batch 维度
    image = image.unsqueeze(0)
    
    return image


# 模型参数
input_height = config.INPUT_SIZE_HIGH
input_width = config.INPUT_SIZE
num_classes = 3

# 标签映射
label_map = {0: 'gold_flash', 1: 'red_flash', 2: 'no_flash'}

# 加载模型
model_path = 'model/best_model_9474_unflatten.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(model_name='conformer', input_height=input_height, input_width=input_width, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 加载数据路径
data_dir = 'data/sorted_images/'
categories = ['gold_flash', 'red_flash', 'no_flash']
image_paths = {category: load_image_paths(os.path.join(data_dir, category)) for category in categories}

# 预测并保存结果
log_file = 'prediction_log.txt'
total_images = 0
correct_predictions = 0
total_time = 0

with open(log_file, 'w') as log:
    for label, category in enumerate(categories):
        paths = image_paths[category]
        for image_path in paths:
            # 加载和预处理图像
            image = cv2.imread(image_path)
            image = preprocess_image(image)
            image = image.to(device)

            # 进行预测并记录耗时
            start_time = time.time()
            with torch.no_grad():
                outputs = model(image)
                score_values = outputs.cpu().numpy().flatten()
                _, predicted = torch.max(outputs, 1)
                predicted_label = predicted.item()
            end_time = time.time()


            # 记录预测时间
            inference_time = end_time - start_time
            total_time += inference_time

            # 记录结果
            log.write(f'{image_path}: {label_map[predicted_label]} \n(scores: gold_flash={score_values[0]:.4f}, red_flash={score_values[1]:.4f}, no_flash={score_values[2]:.4f}, time={inference_time:.4f}s)\n')
            total_images += 1
            if predicted_label == label:
                correct_predictions += 1

# 计算总准确率和平均耗时
accuracy = correct_predictions / total_images
average_time = total_time / total_images

with open(log_file, 'a') as log:
    log.write(f'\nTotal Accuracy: {accuracy * 100:.2f}%\n')
    log.write(f'Average Inference Time: {average_time:.4f}s\n')

print(f'Total Accuracy: {accuracy * 100:.2f}%')
print(f'Average Inference Time: {average_time:.4f}s')