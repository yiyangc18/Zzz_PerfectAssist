import torch
import config
from model import get_model

# 模型参数
input_height = config.INPUT_SIZE_HIGH
input_width = config.INPUT_SIZE
num_classes = 3

# 加载模型
model_path = 'model/best_model_9474_unflatten.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(model_name='conformer', input_height=input_height, input_width=input_width, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 创建虚拟输入
dummy_input = torch.randn(1, 3, input_height, input_width).to(device)

# 转换为ONNX格式
onnx_model_path = 'model/best_model_9474_unflatten.onnx'
torch.onnx.export(model, dummy_input, onnx_model_path, input_names=['input'], output_names=['output'], opset_version=13, verbose=True)
