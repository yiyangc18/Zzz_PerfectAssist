#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// 打印矩阵数据样本 (整数和浮点数)
void print_mat_sample(const cv::Mat& mat, const std::string& step, bool is_float = false) {
    std::cout << step << " Sample:\n";
    if (is_float) {
        std::cout << "(0, 0): [" << mat.at<cv::Vec3f>(0, 0)[0] << ", " 
                                 << mat.at<cv::Vec3f>(0, 0)[1] << ", " 
                                 << mat.at<cv::Vec3f>(0, 0)[2] << "]\n";
        std::cout << "(3, 3): [" << mat.at<cv::Vec3f>(3, 3)[0] << ", " 
                                 << mat.at<cv::Vec3f>(3, 3)[1] << ", " 
                                 << mat.at<cv::Vec3f>(3, 3)[2] << "]\n";
        std::cout << "(5, 5): [" << mat.at<cv::Vec3f>(5, 5)[0] << ", " 
                                 << mat.at<cv::Vec3f>(5, 5)[1] << ", " 
                                 << mat.at<cv::Vec3f>(5, 5)[2] << "]\n";
    } else {
        std::cout << "(0, 0): [" << static_cast<int>(mat.at<cv::Vec3b>(0, 0)[0]) << ", " 
                                 << static_cast<int>(mat.at<cv::Vec3b>(0, 0)[1]) << ", " 
                                 << static_cast<int>(mat.at<cv::Vec3b>(0, 0)[2]) << "]\n";
        std::cout << "(3, 3): [" << static_cast<int>(mat.at<cv::Vec3b>(3, 3)[0]) << ", " 
                                 << static_cast<int>(mat.at<cv::Vec3b>(3, 3)[1]) << ", " 
                                 << static_cast<int>(mat.at<cv::Vec3b>(3, 3)[2]) << "]\n";
        std::cout << "(5, 5): [" << static_cast<int>(mat.at<cv::Vec3b>(5, 5)[0]) << ", " 
                                 << static_cast<int>(mat.at<cv::Vec3b>(5, 5)[1]) << ", " 
                                 << static_cast<int>(mat.at<cv::Vec3b>(5, 5)[2]) << "]\n";
    }
}

cv::Mat preprocess_image(const cv::Mat& image, const cv::Size& target_size) {
    cv::Mat resized_image, float_image;
    // 读取并确保图像是RGB格式
    cv::Mat rgb_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2BGR);
    } else {
        rgb_image = image.clone();
    }
    print_mat_sample(rgb_image, "Step 1 - RGB Image");

    // 缩放图像到目标尺寸
    cv::resize(rgb_image, resized_image, target_size);
    print_mat_sample(resized_image, "Step 2 - Resized Image");

    // 转换为浮点型并归一化
    resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0f);
    print_mat_sample(float_image, "Step 3 - Normalized Image", true);

    // 分离通道并应用标准化参数
    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;
    channels[1] = (channels[1] - 0.456f) / 0.224f;
    channels[2] = (channels[2] - 0.406f) / 0.225f;
    cv::merge(channels, float_image);  // 合并通道
    print_mat_sample(float_image, "Step 4 - Standardized Image", true);

    return float_image;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    // 初始化ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path, session_options);

    // 获取模型输入输出信息
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    std::vector<std::string> input_node_names(num_input_nodes);
    std::vector<std::string> output_node_names(num_output_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name_allocated = session.GetInputNameAllocated(i, allocator);
        input_node_names[i] = input_name_allocated.get();
    }
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name_allocated = session.GetOutputNameAllocated(i, allocator);
        output_node_names[i] = output_name_allocated.get();
    }

    std::vector<const char*> input_node_names_cstr;
    std::vector<const char*> output_node_names_cstr;

    for (const auto& name : input_node_names) {
        input_node_names_cstr.push_back(name.c_str());
    }

    for (const auto& name : output_node_names) {
        output_node_names_cstr.push_back(name.c_str());
    }

    std::vector<int64_t> input_shape = {1, 3, 360, 640};  // 设置输入形状
    size_t input_tensor_size = 1 * 3 * 360 * 640;
    std::vector<float> input_tensor_values(input_tensor_size);

    // 加载并预处理图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    print_mat_sample(image, "Step 1 - Original Image");

    cv::Mat preprocessed_image = preprocess_image(image, cv::Size(640, 360));  // 缩放到(640, 360)

    // 将图像数据转换为输入张量
    std::vector<int64_t> input_shape_with_batch = {1, 3, 360, 640};  // 包含batch维度
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, preprocessed_image.ptr<float>(), input_tensor_size, input_shape_with_batch.data(), input_shape_with_batch.size());

    // 执行推理
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names_cstr.data(), &input_tensor, 1, output_node_names_cstr.data(), 1);
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    // 将输出结果转换为向量
    std::vector<float> logits(output_data, output_data + num_output_nodes);

    // 处理输出结果
    std::cout << "Predicted scores: gold_flash=" << logits[0]
              << ", red_flash=" << logits[1]
              << ", no_flash=" << logits[2] << std::endl;

    return 0;
}
