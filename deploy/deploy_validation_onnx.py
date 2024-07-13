import os
import cv2
import numpy as np
import onnxruntime as ort
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

# 加载ONNX模型
onnx_model_path = 'model/best_model_9474.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 加载数据路径
data_dir = 'data/sorted_images/'
categories = ['gold_flash', 'red_flash', 'no_flash']
image_paths = {category: load_image_paths(os.path.join(data_dir, category)) for category in categories}

# 预测并保存结果
log_file = 'prediction_log.txt'
total_images = 0
correct_predictions_onnx = 0
total_time_onnx = 0

with open(log_file, 'w') as log:
    for label, category in enumerate(categories):
        paths = image_paths[category]
        for image_path in paths:
            # 加载和预处理图像
            image = cv2.imread(image_path)
            image = preprocess_image(image)
            image_np = to_numpy(image)

            # 进行ONNX预测并记录耗时
            start_time = time.time()
            ort_inputs = {ort_session.get_inputs()[0].name: image_np}
            ort_outs = ort_session.run(None, ort_inputs)
            score_values_onnx = np.array(ort_outs).flatten()
            predicted_label_onnx = np.argmax(score_values_onnx)
            end_time = time.time()
            inference_time_onnx = end_time - start_time
            total_time_onnx += inference_time_onnx

            # 记录结果
            log.write(f'{image_path}:\n')
            log.write(f'  ONNX: {label_map[predicted_label_onnx]} (scores: gold_flash={score_values_onnx[0]:.4f}, red_flash={score_values_onnx[1]:.4f}, no_flash={score_values_onnx[2]:.4f}, time={inference_time_onnx:.4f}s)\n')
            total_images += 1
            if predicted_label_onnx == label:
                correct_predictions_onnx += 1

# 计算总准确率和平均耗时
accuracy_onnx = correct_predictions_onnx / total_images
average_time_onnx = total_time_onnx / total_images

with open(log_file, 'a') as log:
    log.write(f'\nTotal Accuracy (ONNX): {accuracy_onnx * 100:.2f}%\n')
    log.write(f'Average Inference Time (ONNX): {average_time_onnx:.4f}s\n')

print(f'Total Accuracy (ONNX): {accuracy_onnx * 100:.2f}%')
print(f'Average Inference Time (ONNX): {average_time_onnx:.4f}s')
