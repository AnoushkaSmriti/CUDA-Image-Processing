#include "convolution.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define TILE_SIZE 16 

__global__ void convolutionKernel(const unsigned char* inputImage, unsigned char* outputImage, 
                                  const float* filter, int imageWidth, int imageHeight, 
                                  int filterWidth) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    int halfFilter = filterWidth / 2;
    
    __shared__ unsigned char tile[TILE_SIZE + 2][TILE_SIZE + 2];
    
    if (row < imageHeight && col < imageWidth) {
        tile[ty + 1][tx + 1] = inputImage[row * imageWidth + col];
        if (tx == 0 && col > 0) tile[ty + 1][0] = inputImage[row * imageWidth + col - 1];
        if (tx == blockDim.x - 1 && col < imageWidth - 1) tile[ty + 1][TILE_SIZE + 1] = inputImage[row * imageWidth + col + 1];
        if (ty == 0 && row > 0) tile[0][tx + 1] = inputImage[(row - 1) * imageWidth + col];
        if (ty == blockDim.y - 1 && row < imageHeight - 1) tile[TILE_SIZE + 1][tx + 1] = inputImage[(row + 1) * imageWidth + col];
    }
    __syncthreads();
    
    if (row < imageHeight && col < imageWidth) {
        float sum = 0.0f;
        for (int i = -halfFilter; i <= halfFilter; i++) {
            for (int j = -halfFilter; j <= halfFilter; j++) {
                int x = tx + 1 + j;
                int y = ty + 1 + i;
                sum += tile[y][x] * filter[(i + halfFilter) * filterWidth + (j + halfFilter)];
            }
        }
        outputImage[row * imageWidth + col] = min(max(int(sum), 0), 255);
    }
}

void applyConvolution(const unsigned char* h_inputImage, unsigned char* h_outputImage, 
                      const float* h_filter, int imageWidth, int imageHeight, int filterWidth) {
    unsigned char *d_inputImage, *d_outputImage;
    float *d_filter;
    int imageSize = imageWidth * imageHeight * sizeof(unsigned char);
    int filterSize = filterWidth * filterWidth * sizeof(float);
    
    cudaMalloc((void**)&d_inputImage, imageSize);
    cudaMalloc((void**)&d_outputImage, imageSize);
    cudaMalloc((void**)&d_filter, filterSize);
    
    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((imageWidth + TILE_SIZE - 1) / TILE_SIZE, 
                  (imageHeight + TILE_SIZE - 1) / TILE_SIZE);
    
    convolutionKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, d_filter, imageWidth, imageHeight, filterWidth);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_filter);
}