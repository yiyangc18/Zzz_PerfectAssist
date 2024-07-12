import cv2
import numpy as np

def preprocess_image(image, target_size=(416, 234)):
    # 确保图像是单通道灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 缩放图像
    image = cv2.resize(image, target_size)
    
    # 创建一个三通道图像以便于后续处理
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    result_orange = cv2.bitwise_and(image, image, mask=mask_orange)
    result_red = cv2.bitwise_and(image, image, mask=mask_red)

    result_orange[result_orange > 0] = 168
    result_red[result_red > 0] = 255

    combined_gray = cv2.addWeighted(result_orange, 1, result_red, 1, 0)

    kernel = np.ones((2, 2), np.uint8)
    combined_gray = cv2.dilate(combined_gray, kernel, iterations=1)
    combined_gray = cv2.erode(combined_gray, kernel, iterations=1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(combined_gray)

    return enhanced_gray

def preprocess_image(image, target_size=(416, 234)):
    # 确保图像是单通道灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 缩放图像
    image = cv2.resize(image, target_size)
    
    # 创建一个三通道图像以便于后续处理
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    result_orange = cv2.bitwise_and(image, image, mask=mask_orange)
    result_red = cv2.bitwise_and(image, image, mask=mask_red)

    result_orange[result_orange > 0] = 128
    result_red[result_red > 0] = 255

    combined_gray = cv2.addWeighted(result_orange, 1, result_red, 1, 0)

#     kernel = np.ones((2, 2), np.uint8)
#     combined_gray = cv2.dilate(combined_gray, kernel, iterations=1)
#     combined_gray = cv2.erode(combined_gray, kernel, iterations=1)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_gray = clahe.apply(combined_gray)

    return combined_gray

def preprocess_image_onlyresize(image, target_size=(416, 234)):
    # 确保图像是RGB图像
    if len(image.shape) == 2:  # 如果是灰度图
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 缩放图像
    image = cv2.resize(image, target_size)
    
    return image
