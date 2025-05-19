#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include <cuda_runtime.h>
#include "utils.cuh"

// Kernel for 2D convolution using shared memory
__global__ void convolutionKernel(const unsigned char* inputImage, unsigned char* outputImage, 
                                  const float* filter, int imageWidth, int imageHeight, 
                                  int filterWidth);

// Host function to call the convolution kernel
void applyConvolution(const unsigned char* h_inputImage, unsigned char* h_outputImage, 
                      const float* h_filter, int imageWidth, int imageHeight, int filterWidth);

#endif
