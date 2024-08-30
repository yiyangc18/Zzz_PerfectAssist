import onnx
import onnxoptimizer

# 加载 ONNX 模型
model = onnx.load('model/onnx_model_simplified.onnx')

# 指定要应用的优化策略
passes = ['eliminate_nop_transpose', 'fuse_bn_into_conv']

# 优化模型
optimized_model = onnxoptimizer.optimize(model, passes)  # 使用别名 opt

# 保存优化后的模型
onnx.save(optimized_model, 'model/9474_simplified_optimized.onnx')

print("Model optimized successfully.")
