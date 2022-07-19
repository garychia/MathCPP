#include "CUDAUtilities.cuh"

#include <iostream>

void CheckCUDAStatus(const cudaError_t &error)
{
    if (error != cudaSuccess)
    {
        std::printf("Error: %s:%d, ", __FILE__, __LINE__);
        std::printf("code:%d, reason: %sn", error, cudaGetErrorString(error));
        std::exit(1);
    }
}