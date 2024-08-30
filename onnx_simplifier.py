import onnx
from onnxsim import simplify

# 加载 ONNX 模型
model = onnx.load('model/best_model_9474.onnx')

# 简化模型
model_simp, check = simplify(model)

# 保存简化后的模型
onnx.save(model_simp, 'model/onnx_model_simplified.onnx')

print("Model simplified successfully.")
