#include "convolution.cuh"
#include "morphology2.cuh"
#include "utils.cuh"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

// Convolution kernel (defined here to fix undefined identifier error)
__global__ void convolutionKernelConfigurable(const unsigned char* inputImage, unsigned char* outputImage,
    const float* filter, int imageWidth, int imageHeight,
    int filterWidth, LaunchConfig config) {
    int halfFilter = filterWidth / 2;

    if (config == LaunchConfig::ROW_PER_THREAD) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= imageHeight) return;

        for (int col = 0; col < imageWidth; col++) {
            float sum = 0.0f;
            for (int i = -halfFilter; i <= halfFilter; i++) {
                for (int j = -halfFilter; j <= halfFilter; j++) {
                    int x = col + j;
                    int y = row + i;
                    if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight) {
                        sum += inputImage[y * imageWidth + x] *
                            filter[(i + halfFilter) * filterWidth + (j + halfFilter)];
                    }
                }
            }
            outputImage[row * imageWidth + col] = min(max(int(sum), 0), 255);
        }
    }
    else if (config == LaunchConfig::COL_PER_THREAD) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= imageWidth) return;

        for (int row = 0; row < imageHeight; row++) {
            float sum = 0.0f;
            for (int i = -halfFilter; i <= halfFilter; i++) {
                for (int j = -halfFilter; j <= halfFilter; j++) {
                    int x = col + j;
                    int y = row + i;
                    if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight) {
                        sum += inputImage[y * imageWidth + x] *
                            filter[(i + halfFilter) * filterWidth + (j + halfFilter)];
                    }
                }
            }
            outputImage[row * imageWidth + col] = min(max(int(sum), 0), 255);
        }
    }
    else { // PIXEL_PER_THREAD
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < imageHeight && col < imageWidth) {
            float sum = 0.0f;
            for (int i = -halfFilter; i <= halfFilter; i++) {
                for (int j = -halfFilter; j <= halfFilter; j++) {
                    int x = col + j;
                    int y = row + i;
                    if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight) {
                        sum += inputImage[y * imageWidth + x] *
                            filter[(i + halfFilter) * filterWidth + (j + halfFilter)];
                    }
                }
            }
            outputImage[row * imageWidth + col] = min(max(int(sum), 0), 255);
        }
    }
}

// Generic wrapper for convolution
void gpuConvolution(const uchar* h_input, uchar* h_output, int width, int height,
    const float* h_kernel, int kernelSize, LaunchConfig config) {
    uchar* d_input, * d_output;
    float* d_kernel;
    size_t imageSize = width * height * sizeof(uchar);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    if (config == LaunchConfig::ROW_PER_THREAD) {
        dim3 blockSize(256);
        dim3 gridSize((height + blockSize.x - 1) / blockSize.x);
        convolutionKernelConfigurable << <gridSize, blockSize >> > (d_input, d_output,
            d_kernel, width, height,
            kernelSize, config);
    }
    else if (config == LaunchConfig::COL_PER_THREAD) {
        dim3 blockSize(256);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x);
        convolutionKernelConfigurable << <gridSize, blockSize >> > (d_input, d_output,
            d_kernel, width, height,
            kernelSize, config);
    }
    else { // PIXEL_PER_THREAD
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
            (height + blockSize.y - 1) / blockSize.y);
        convolutionKernelConfigurable << <gridSize, blockSize >> > (d_input, d_output,
            d_kernel, width, height,
            kernelSize, config);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

// Performance testing function
void compareOperations(const Mat& inputImage, int kernelSize) {
    Mat cpuOutput(inputImage.size(), CV_8UC1);
    Mat gpuOutputs[3][3]; // [operation][config]
    vector<float> convKernel(kernelSize * kernelSize, 1.0f / (kernelSize * kernelSize));
    vector<int> morphKernel(kernelSize * kernelSize, 1);

    LaunchConfig configs[] = { LaunchConfig::ROW_PER_THREAD,
                            LaunchConfig::COL_PER_THREAD,
                            LaunchConfig::PIXEL_PER_THREAD };
    string configNames[] = { "RowPerThread", "ColPerThread", "PixelPerThread" };
    string opNames[] = { "Convolution", "Erosion", "Dilation" };

    // CPU Operations
    auto start = chrono::high_resolution_clock::now();
    filter2D(inputImage, cpuOutput, -1, Mat(kernelSize, kernelSize, CV_32F, convKernel.data()));
    auto end = chrono::high_resolution_clock::now();
    cout << "CPU Convolution Time: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    imwrite("cpu_convolution.png", cpuOutput);

    start = chrono::high_resolution_clock::now();
    cpu_erosion(inputImage, cpuOutput, morphKernel);
    end = chrono::high_resolution_clock::now();
    cout << "CPU Erosion Time: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    imwrite("cpu_erosion.png", cpuOutput);

    start = chrono::high_resolution_clock::now();
    cpu_dilation(inputImage, cpuOutput, morphKernel);
    end = chrono::high_resolution_clock::now();
    cout << "CPU Dilation Time: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    imwrite("cpu_dilation.png", cpuOutput);

    // GPU Operations
    for (int i = 0; i < 3; i++) { // Operations
        for (int j = 0; j < 3; j++) { // Configurations
            gpuOutputs[i][j] = Mat(inputImage.size(), CV_8UC1);
            start = chrono::high_resolution_clock::now();
            if (i == 0) { // Convolution
                gpuConvolution(inputImage.data, gpuOutputs[i][j].data,
                    inputImage.cols, inputImage.rows, convKernel.data(),
                    kernelSize, configs[j]);
            }
            else if (i == 1) { // Erosion
                gpu_erosion_configurable(inputImage.data, gpuOutputs[i][j].data,
                    inputImage.cols, inputImage.rows, morphKernel.data(),
                    kernelSize, configs[j]);
            }
            else { // Dilation
                gpu_dilation_configurable(inputImage.data, gpuOutputs[i][j].data,
                    inputImage.cols, inputImage.rows, morphKernel.data(),
                    kernelSize, configs[j]);
            }
            end = chrono::high_resolution_clock::now();
            cout << "GPU " << opNames[i] << " (" << configNames[j] << ") Time: "
                << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
            imwrite("gpu_" + opNames[i] + "_" + configNames[j] + ".png", gpuOutputs[i][j]);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    Mat inputImage = imread(argv[1], IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        cerr << "Error: Could not open image!" << endl;
        return -1;
    }

    int kernelSize = 3;
    compareOperations(inputImage, kernelSize);

    cout << "Processing complete. Output images saved." << endl;
    return 0;
}