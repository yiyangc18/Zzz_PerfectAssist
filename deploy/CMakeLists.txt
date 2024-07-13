cmake_minimum_required(VERSION 3.10)
project(ONNXRuntimeDemo)

set(CMAKE_CXX_STANDARD 14)  # 将标准设置为C++14或更高
set(ONNXRUNTIME_DIR /home/chery/code/Zzz_PerfectAssist/onnx_deploy/onnxruntime-linux-x64-gpu-1.18.1)  # 替换为你的ONNX Runtime库路径

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_DIR}/include /usr/include/opencv4/opencv /usr/include/opencv4)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(onnx_inference onnx_deploy_test.cpp)
target_link_libraries(onnx_inference ${OpenCV_LIBS} ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
