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


