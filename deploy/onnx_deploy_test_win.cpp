#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <locale>
#include <codecvt>

// 预处理函数
cv::Mat preprocess_image(const cv::Mat& image, const cv::Size& target_size) {
    cv::Mat resized_image, float_image;
    cv::resize(image, resized_image, target_size);
    resized_image.convertTo(float_image, CV_32F, 1.0 / 255);
    cv::Mat channels[3];
    cv::split(float_image, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, float_image);
    return float_image;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    // 将模型路径转换为宽字符
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring model_path_w = converter.from_bytes(model_path);

    // 初始化ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 构造Session对象
    Ort::Session session(env, model_path_w.c_str(), session_options);

    // 获取模型输入输出信息
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputNameAllocated(0, allocator).get();
    const char* output_name = session.GetOutputNameAllocated(0, allocator).get();

    std::vector<int64_t> input_shape = {1, 3, 234, 416};
    size_t input_tensor_size = 1 * 3 * 234 * 416;
    std::vector<float> input_tensor_values(input_tensor_size);

    // 加载并预处理图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    cv::Mat preprocessed_image = preprocess_image(image, cv::Size(416, 234));

    // 将图像数据转换为输入张量
    std::memcpy(input_tensor_values.data(), preprocessed_image.data, input_tensor_size * sizeof(float));
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    // 执行推理
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    // 处理输出结果
    std::cout << "Predicted scores: gold_flash=" << output_data[0]
              << ", red_flash=" << output_data[1]
              << ", no_flash=" << output_data[2] << std::endl;

    return 0;
}
