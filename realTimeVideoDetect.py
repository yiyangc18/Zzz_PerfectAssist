import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import mss
import config  
import threading
import queue
import concurrent.futures
import torchvision.transforms as transforms
from screen_show import OverlayText

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

class RealTimeProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.preprocess_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.preprocess_times = []
        self.prediction_times = []
        self.ort_session = self.initialize_model()
        self.transform = self.get_transform()
        self.overlay = OverlayText()
        self.monitor = mss.mss().monitors[1]

    def get_transform(self):
        # 定义转换对象，只创建一次
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def initialize_model(self):
        return ort.InferenceSession(self.model_path)

    def preprocess_image(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 直接在屏幕捕捉时指定目标分辨率 (640, 360)，避免 resize 操作
        # 如果必须 resize:
        image = cv2.resize(image, (640, 360))

        # 将图像转换为 float32 并进行归一化，同时进行通道转换
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std

        # 直接转换格式为 (C, H, W) 并添加批次维度
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0)

        return image  # 已经是 float32 类型

    def screen_capture_preprocess(self):
        with mss.mss() as sct:
            while not self.stop_event.is_set():
                start_time = time.time()
                screen = np.array(sct.grab(self.monitor))
                frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                input_tensor = self.preprocess_image(frame)
                self.preprocess_queue.put(input_tensor)
                self.preprocess_times.append(time.time() - start_time)
                time.sleep(0.01)  # 控制捕获频率
                print("average preprocess_time: ", sum(self.preprocess_times) / len(self.preprocess_times))

    def get_prediction(self, input_tensor):
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return np.argmax(ort_outs[0])

    def inference_function(self):
        while not self.stop_event.is_set():
            try:
                input_tensor = self.preprocess_queue.get(block=True, timeout=1)
                pre_start_time = time.time()
                class_result = self.get_prediction(input_tensor)
                class_name = class_names.get(class_result, "Unknown")
                text_color = class_colors.get(class_result, (255, 255, 255))
                self.result_queue.put((class_name, text_color))
                self.prediction_times.append(time.time() - pre_start_time)
                print("average prediction_time: ", sum(self.prediction_times) / len(self.prediction_times))
            except queue.Empty:
                continue

    def start(self):
        capture_thread = threading.Thread(target=self.screen_capture_preprocess)
        inference_thread = threading.Thread(target=self.inference_function)

        capture_thread.start()
        inference_thread.start()

        try:
            while not self.stop_event.is_set():
                try:
                    class_name, text_color = self.result_queue.get(block=True, timeout=1)
                    self.overlay.update_text(class_name, text_color)
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            self.stop_event.set()
            capture_thread.join()
            inference_thread.join()

if __name__ == "__main__":
    model_path = config.ONXX_MODEL_PATH
    processor = RealTimeProcessor(model_path)
    processor.start()
