#ifndef MORPHOLOGY_CUH
#define MORPHOLOGY_CUH

#include <opencv2/opencv.hpp>
#include <vector>

enum class LaunchConfig {
    ROW_PER_THREAD,
    COL_PER_THREAD,
    PIXEL_PER_THREAD
};

void cpu_erosion(const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<int>& kernel);
void cpu_dilation(const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<int>& kernel);

void gpu_erosion_configurable(const uchar* input, uchar* output, int width, int height,
    const int* kernel, int kernelSize, LaunchConfig config);
void gpu_dilation_configurable(const uchar* input, uchar* output, int width, int height,
    const int* kernel, int kernelSize, LaunchConfig config);

#endif