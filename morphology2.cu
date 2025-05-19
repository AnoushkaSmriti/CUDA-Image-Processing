#include "morphology2.cuh"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// CPU implementation of erosion (unchanged)
void cpu_erosion(const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<int>& kernel) {
    int kernelSize = 3;
    int offset = kernelSize / 2;

    for (int i = offset; i < inputImage.rows - offset; ++i) {
        for (int j = offset; j < inputImage.cols - offset; ++j) {
            uchar minValue = 255;
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    int ni = i + ki - offset;
                    int nj = j + kj - offset;
                    if (kernel[ki * kernelSize + kj] == 1) {
                        minValue = std::min(minValue, inputImage.at<uchar>(ni, nj));
                    }
                }
            }
            outputImage.at<uchar>(i, j) = minValue;
        }
    }
}

// CPU implementation of dilation (unchanged)
void cpu_dilation(const cv::Mat& inputImage, cv::Mat& outputImage, const std::vector<int>& kernel) {
    int kernelSize = 3;
    int offset = kernelSize / 2;

    for (int i = offset; i < inputImage.rows - offset; ++i) {
        for (int j = offset; j < inputImage.cols - offset; ++j) {
            uchar maxValue = 0;
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    int ni = i + ki - offset;
                    int nj = j + kj - offset;
                    if (kernel[ki * kernelSize + kj] == 1) {
                        maxValue = std::max(maxValue, inputImage.at<uchar>(ni, nj));
                    }
                }
            }
            outputImage.at<uchar>(i, j) = maxValue;
        }
    }
}

// GPU kernel for erosion (configurable)
__global__ void gpu_erosion_kernel_configurable(const uchar* input, uchar* output,
    int width, int height, const int* kernel,
    int kernelSize, LaunchConfig config) {
    int offset = kernelSize / 2;

    if (config == LaunchConfig::ROW_PER_THREAD) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < offset || row >= height - offset) return;

        for (int x = offset; x < width - offset; x++) {
            uchar minValue = 255;
            for (int ki = -offset; ki <= offset; ++ki) {
                for (int kj = -offset; kj <= offset; ++kj) {
                    int ni = row + ki;
                    int nj = x + kj;
                    if (kernel[(ki + offset) * kernelSize + (kj + offset)] == 1) {
                        minValue = min(minValue, input[ni * width + nj]);
                    }
                }
            }
            output[row * width + x] = minValue;
        }
    }
    else if (config == LaunchConfig::COL_PER_THREAD) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col < offset || col >= width - offset) return;

        for (int y = offset; y < height - offset; y++) {
            uchar minValue = 255;
            for (int ki = -offset; ki <= offset; ++ki) {
                for (int kj = -offset; kj <= offset; ++kj) {
                    int ni = y + ki;
                    int nj = col + kj;
                    if (kernel[(ki + offset) * kernelSize + (kj + offset)] == 1) {
                        minValue = min(minValue, input[ni * width + nj]);
                    }
                }
            }
            output[y * width + col] = minValue;
        }
    }
    else { // PIXEL_PER_THREAD
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= offset && x < width - offset && y >= offset && y < height - offset) {
            uchar minValue = 255;
            for (int ki = -offset; ki <= offset; ++ki) {
                for (int kj = -offset; kj <= offset; ++kj) {
                    int ni = y + ki;
                    int nj = x + kj;
                    if (kernel[(ki + offset) * kernelSize + (kj + offset)] == 1) {
                        minValue = min(minValue, input[ni * width + nj]);
                    }
                }
            }
            output[y * width + x] = minValue;
        }
    }
}

// GPU kernel for dilation (configurable)
__global__ void gpu_dilation_kernel_configurable(const uchar* input, uchar* output,
    int width, int height, const int* kernel,
    int kernelSize, LaunchConfig config) {
    int offset = kernelSize / 2;

    if (config == LaunchConfig::ROW_PER_THREAD) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < offset || row >= height - offset) return;

        for (int x = offset; x < width - offset; x++) {
            uchar maxValue = 0;
            for (int ki = -offset; ki <= offset; ++ki) {
                for (int kj = -offset; kj <= offset; ++kj) {
                    int ni = row + ki;
                    int nj = x + kj;
                    if (kernel[(ki + offset) * kernelSize + (kj + offset)] == 1) {
                        maxValue = max(maxValue, input[ni * width + nj]);
                    }
                }
            }
            output[row * width + x] = maxValue;
        }
    }
    else if (config == LaunchConfig::COL_PER_THREAD) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col < offset || col >= width - offset) return;

        for (int y = offset; y < height - offset; y++) {
            uchar maxValue = 0;
            for (int ki = -offset; ki <= offset; ++ki) {
                for (int kj = -offset; kj <= offset; ++kj) {
                    int ni = y + ki;
                    int nj = col + kj;
                    if (kernel[(ki + offset) * kernelSize + (kj + offset)] == 1) {
                        maxValue = max(maxValue, input[ni * width + nj]);
                    }
                }
            }
            output[y * width + col] = maxValue;
        }
    }
    else { // PIXEL_PER_THREAD
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= offset && x < width - offset && y >= offset && y < height - offset) {
            uchar maxValue = 0;
            for (int ki = -offset; ki <= offset; ++ki) {
                for (int kj = -offset; kj <= offset; ++kj) {
                    int ni = y + ki;
                    int nj = x + kj;
                    if (kernel[(ki + offset) * kernelSize + (kj + offset)] == 1) {
                        maxValue = max(maxValue, input[ni * width + nj]);
                    }
                }
            }
            output[y * width + x] = maxValue;
        }
    }
}

// GPU erosion wrapper
void gpu_erosion_configurable(const uchar* input, uchar* output, int width, int height,
    const int* kernel, int kernelSize, LaunchConfig config) {
    uchar* d_input, * d_output;
    int* d_kernel;
    size_t imageSize = width * height * sizeof(uchar);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(int);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    if (config == LaunchConfig::ROW_PER_THREAD) {
        dim3 blockSize(256);
        dim3 gridSize((height + blockSize.x - 1) / blockSize.x);
        gpu_erosion_kernel_configurable << <gridSize, blockSize >> > (d_input, d_output,
            width, height, d_kernel,
            kernelSize, config);
    }
    else if (config == LaunchConfig::COL_PER_THREAD) {
        dim3 blockSize(256);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x);
        gpu_erosion_kernel_configurable << <gridSize, blockSize >> > (d_input, d_output,
            width, height, d_kernel,
            kernelSize, config);
    }
    else { // PIXEL_PER_THREAD
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
            (height + blockSize.y - 1) / blockSize.y);
        gpu_erosion_kernel_configurable << <gridSize, blockSize >> > (d_input, d_output,
            width, height, d_kernel,
            kernelSize, config);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

// GPU dilation wrapper
void gpu_dilation_configurable(const uchar* input, uchar* output, int width, int height,
    const int* kernel, int kernelSize, LaunchConfig config) {
    uchar* d_input, * d_output;
    int* d_kernel;
    size_t imageSize = width * height * sizeof(uchar);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(int);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    if (config == LaunchConfig::ROW_PER_THREAD) {
        dim3 blockSize(256);
        dim3 gridSize((height + blockSize.x - 1) / blockSize.x);
        gpu_dilation_kernel_configurable << <gridSize, blockSize >> > (d_input, d_output,
            width, height, d_kernel,
            kernelSize, config);
    }
    else if (config == LaunchConfig::COL_PER_THREAD) {
        dim3 blockSize(256);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x);
        gpu_dilation_kernel_configurable << <gridSize, blockSize >> > (d_input, d_output,
            width, height, d_kernel,
            kernelSize, config);
    }
    else { // PIXEL_PER_THREAD
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
            (height + blockSize.y - 1) / blockSize.y);
        gpu_dilation_kernel_configurable << <gridSize, blockSize >> > (d_input, d_output,
            width, height, d_kernel,
            kernelSize, config);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}