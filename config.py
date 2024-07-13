# config.py


GOLDFLASH_IMAGE_DIR = r'data/sorted_images/gold_flash'
RED_FLASH_IMAGE_DIR = r'data/sorted_images/red_flash'
NO_FLASH_IMAGE_DIR = r'data/sorted_images/no_flash'

# 训练参数设置
BATCH_SIZE = 2
NUM_EPOCHS = 30
LEARNING_RATE = 0.0005
SHUFFLE = True
WARMUP_PERCENT = 0.3  # warmup 百分比，表示前 % 的步骤用于 warmup


# 模型参数设置
NUM_CLASSES = 3  # 类别数
INPUT_SIZE = 640  # 输入图像大小
INPUT_SIZE_HIGH = 360  # 输入图像大小


# 实时画面采集运行参数设置
MAX_FRAME = 30  # 最大采集帧数 运行帧数太高占CPU
SAME_PREDICTION_THRESHOLD = 5 # 连续相同预测-触发阈值帧数
# red flash 键盘输入 默认Shift键
# RED_INPUT = 0x10
# gold flash 键盘输入 默认Spacebar键
# GOLD_INPUT = 0x20

# red flash 键盘输入Q键
RED_INPUT = 0x51
# gold flash 键盘输入E键
GOLD_INPUT = 0x45

PTH_MODEL_PATH = 'model/best_model_9474_unflatten.pth'
ONXX_MODEL_PATH = 'model/best_model_9474.onnx'