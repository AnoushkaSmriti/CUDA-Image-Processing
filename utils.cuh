4#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void* cudaMallocSafe(size_t size) {
    void* devPtr;
    checkCudaError(cudaMalloc(&devPtr, size), "cudaMalloc failed");
    return devPtr;
}

inline void cudaMemcpySafe(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    checkCudaError(cudaMemcpy(dst, src, size, kind), "cudaMemcpy failed");
}

inline void cudaFreeSafe(void* devPtr) {
    checkCudaError(cudaFree(devPtr), "cudaFree failed");
}

struct Timer {
    std::chrono::high_resolution_clock::time_point start, end;
    void startTimer() { start = std::chrono::high_resolution_clock::now(); }
    void stopTimer(const char* operation) {
        end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << operation << " Execution Time: " << duration << " ms" << std::endl;
    }
};

#endif