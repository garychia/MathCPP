#include "ArithmeticOperations.cuh"

#include "GPUArray.cuh"
#include "CUDAHelpers.hpp"
#include "CUDAUtilities.cuh"

#define MIN(a, b) (a) < (b) ? (a) : (b)

#define FUNCTION_IMPLEMENTATIONS_WITH_THREE_TYPES(imple_macro, ...)                                      \
    EXPAND(imple_macro(int, int, int, __VA_ARGS__));                                                     \
    EXPAND(imple_macro(float, float, float, __VA_ARGS__));                                               \
    EXPAND(imple_macro(double, double, double, __VA_ARGS__));                                            \
    EXPAND(imple_macro(std::size_t, std::size_t, std::size_t, __VA_ARGS__));                             \
    EXPAND(TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, int, float, __VA_ARGS__))                   \
    EXPAND(TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, int, double, __VA_ARGS__))                  \
    EXPAND(TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, int, std::size_t, __VA_ARGS__))             \
    EXPAND(TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, float, double, __VA_ARGS__))                \
    EXPAND(TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, float, std::size_t, __VA_ARGS__))           \
    EXPAND(TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, double, std::size_t, __VA_ARGS__))          \
    EXPAND(THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, int, float, std::size_t, __VA_ARGS__))    \
    EXPAND(THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, int, std::size_t, double, __VA_ARGS__))   \
    EXPAND(THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, std::size_t, float, double, __VA_ARGS__)) \
    EXPAND(THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(imple_macro, int, float, double, __VA_ARGS__))

#define GPU_ARRAY_POPULATE_FUNCTION_IMPLEMENTATION(func_name, arr_type, helper_func)      \
    void func_name(arr_type *arr, std::size_t size, arr_type value)                       \
    {                                                                                     \
        const std::size_t threadsPerBlock = size > 32 ? 32 : size;                        \
        const std::size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock; \
        helper_func<<<blocksPerGrid, threadsPerBlock>>>(arr, size, value);                \
    }

#define GPU_ARRAY_POPULATE_FUNCTION_IMPLEMENTATIONS(func_name, helper_func)   \
    GPU_ARRAY_POPULATE_FUNCTION_IMPLEMENTATION(func_name, int, helper_func)   \
    GPU_ARRAY_POPULATE_FUNCTION_IMPLEMENTATION(func_name, float, helper_func) \
    GPU_ARRAY_POPULATE_FUNCTION_IMPLEMENTATION(func_name, double, helper_func)

#define TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(output_type, op_type_1, op_type_2, func_name, helper_func) \
    void func_name(output_type *dest, const op_type_1 *operand1, const op_type_2 *operand2, std::size_t size)     \
    {                                                                                                             \
        const std::size_t bytesOfOutput = size * sizeof(output_type);                                             \
        const std::size_t bytesOfArray1 = size * sizeof(op_type_1);                                               \
        const std::size_t bytesOfArray2 = size * sizeof(op_type_2);                                               \
        output_type *gpuDest;                                                                                     \
        CheckCUDAStatus(cudaMalloc(&gpuDest, bytesOfOutput));                                                     \
        op_type_1 *op1;                                                                                           \
        CheckCUDAStatus(cudaMalloc(&op1, bytesOfArray1));                                                         \
        op_type_2 *op2;                                                                                           \
        CheckCUDAStatus(cudaMalloc(&op2, bytesOfArray2));                                                         \
        const std::size_t numOfStreams = 8;                                                                       \
        const std::size_t arrayChunckSize = (size + numOfStreams - 1) / numOfStreams;                             \
        cudaStream_t streams[numOfStreams];                                                                       \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                            \
            CheckCUDAStatus(cudaStreamCreate(&streams[i]));                                                       \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                            \
        {                                                                                                         \
            const auto lowerBound = i * arrayChunckSize;                                                          \
            const auto upperBound = MIN(lowerBound + arrayChunckSize, size);                                      \
            const auto nElements = upperBound - lowerBound;                                                       \
            if (0 == nElements)                                                                                   \
                break;                                                                                            \
            CheckCUDAStatus(cudaMemcpyAsync(gpuDest + lowerBound, dest + lowerBound,                              \
                                            sizeof(output_type) * nElements, cudaMemcpyHostToDevice,              \
                                            streams[i]));                                                         \
            CheckCUDAStatus(cudaMemcpyAsync(op1 + lowerBound, operand1 + lowerBound,                              \
                                            sizeof(op_type_1) * nElements, cudaMemcpyHostToDevice,                \
                                            streams[i]));                                                         \
            CheckCUDAStatus(cudaMemcpyAsync(op2 + lowerBound, operand2 + lowerBound,                              \
                                            sizeof(op_type_2) * nElements, cudaMemcpyHostToDevice,                \
                                            streams[i]));                                                         \
            const std::size_t threadsPerBlock = nElements > 32 ? 32 : nElements;                                  \
            const std::size_t blocksPerGrid = (nElements + threadsPerBlock - 1) / threadsPerBlock;                \
            helper_func<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(                                       \
                gpuDest + lowerBound,                                                                             \
                op1 + lowerBound,                                                                                 \
                op2 + lowerBound,                                                                                 \
                nElements);                                                                                       \
            CheckCUDAStatus(cudaMemcpyAsync(dest + lowerBound, gpuDest + lowerBound,                              \
                                            sizeof(output_type) * nElements, cudaMemcpyDeviceToHost,              \
                                            streams[i]));                                                         \
        }                                                                                                         \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                            \
            CheckCUDAStatus(cudaStreamSynchronize(streams[i]));                                                   \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                            \
            CheckCUDAStatus(cudaStreamDestroy(streams[i]));                                                       \
        CheckCUDAStatus(cudaFree(gpuDest));                                                                       \
        CheckCUDAStatus(cudaFree(op1));                                                                           \
        CheckCUDAStatus(cudaFree(op2));                                                                           \
    }

#define TWO_GPU_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(output_type, op_type_1, op_type_2, func_name, helper_func) \
    void func_name(output_type *dest, const op_type_1 *operand1, const op_type_2 *operand2, std::size_t size)         \
    {                                                                                                                 \
        const std::size_t threadsPerBlock = size > 32 ? 32 : size;                                                    \
        const std::size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;                             \
        helper_func<<<blocksPerGrid, threadsPerBlock>>>(dest, operand1, operand2, size);                              \
    }

#define TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(func_name, helper_func) \
    FUNCTION_IMPLEMENTATIONS_WITH_THREE_TYPES(TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION, func_name, helper_func)

#define TWO_GPU_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(func_name, helper_func) \
    FUNCTION_IMPLEMENTATIONS_WITH_THREE_TYPES(TWO_GPU_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION, func_name, helper_func)

#define ARRAY_SCALER_FUNCTION_IMPLEMENTATION(output_type, arr_type, scaler_type, func_name, helper_func) \
    void func_name(output_type *dest, const arr_type *arr, const scaler_type scaler, std::size_t size)   \
    {                                                                                                    \
        const std::size_t bytesOfOutput = size * sizeof(output_type);                                    \
        const std::size_t bytesOfArray1 = size * sizeof(arr_type);                                       \
        output_type *destGPU;                                                                            \
        cudaMalloc(&destGPU, bytesOfOutput);                                                             \
        arr_type *arrGPU;                                                                                \
        cudaMalloc(&arrGPU, bytesOfArray1);                                                              \
        const std::size_t numOfStreams = 8;                                                              \
        const std::size_t arrayChunckSize = (size + numOfStreams - 1) / numOfStreams;                    \
        cudaStream_t streams[numOfStreams];                                                              \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                   \
            cudaStreamCreate(&streams[i]);                                                               \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                   \
        {                                                                                                \
            const auto lowerBound = i * arrayChunckSize;                                                 \
            const auto upperBound = MIN(lowerBound + arrayChunckSize, size);                             \
            const auto nElements = upperBound - lowerBound;                                              \
            if (0 == nElements)                                                                          \
                break;                                                                                   \
            cudaMemcpyAsync(destGPU + lowerBound, dest + lowerBound,                                     \
                            sizeof(output_type) * nElements, cudaMemcpyHostToDevice,                     \
                            streams[i]);                                                                 \
            cudaMemcpyAsync(arrGPU + lowerBound, arr + lowerBound,                                       \
                            sizeof(arr_type) * nElements, cudaMemcpyHostToDevice,                        \
                            streams[i]);                                                                 \
            const std::size_t threadsPerBlock = nElements > 32 ? 32 : nElements;                         \
            const std::size_t blocksPerGrid = (nElements + threadsPerBlock - 1) / threadsPerBlock;       \
            helper_func<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(                              \
                destGPU + lowerBound,                                                                    \
                arrGPU + lowerBound,                                                                     \
                scaler,                                                                                  \
                nElements);                                                                              \
            cudaMemcpyAsync(dest + lowerBound, destGPU + lowerBound,                                     \
                            sizeof(output_type) * nElements, cudaMemcpyDeviceToHost,                     \
                            streams[i]);                                                                 \
        }                                                                                                \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                   \
            cudaStreamSynchronize(streams[i]);                                                           \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                   \
            cudaStreamDestroy(streams[i]);                                                               \
        cudaFree(destGPU);                                                                               \
        cudaFree(arrGPU);                                                                                \
    }

#define GPU_ARRAY_SCALER_FUNCTION_IMPLEMENTATION(output_type, arr_type, scaler_type, func_name, helper_func) \
    void func_name(output_type *dest, const arr_type *arr, const scaler_type scaler, std::size_t size)       \
    {                                                                                                        \
        const std::size_t threadsPerBlock = size > 32 ? 32 : size;                                           \
        const std::size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;                    \
        helper_func<<<blocksPerGrid, threadsPerBlock>>>(dest, arr, scaler, size);                            \
    }

#define ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(func_name, helper_func) \
    FUNCTION_IMPLEMENTATIONS_WITH_THREE_TYPES(ARRAY_SCALER_FUNCTION_IMPLEMENTATION, func_name, helper_func)

#define GPU_ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(func_name, helper_func) \
    FUNCTION_IMPLEMENTATIONS_WITH_THREE_TYPES(GPU_ARRAY_SCALER_FUNCTION_IMPLEMENTATION, func_name, helper_func)

#define SCALER_ARRAY_FUNCTION_IMPLEMENTATION(output_type, scaler_type, arr_type, func_name, op)        \
    void func_name(output_type *dest, const scaler_type scaler, const arr_type *arr, std::size_t size) \
    {                                                                                                  \
        const std::size_t bytesOfOutput = size * sizeof(output_type);                                  \
        const std::size_t bytesOfArray1 = size * sizeof(arr_type);                                     \
        output_type *destGPU;                                                                          \
        cudaMalloc(&destGPU, bytesOfOutput);                                                           \
        arr_type *arrGPU;                                                                              \
        cudaMalloc(&arrGPU, bytesOfArray1);                                                            \
        const std::size_t numOfStreams = 8;                                                            \
        const std::size_t arrayChunckSize = (size + numOfStreams - 1) / numOfStreams;                  \
        cudaStream_t streams[numOfStreams];                                                            \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                 \
            cudaStreamCreate(&streams[i]);                                                             \
        const auto f = [=] __device__(const arr_type &e) { return scaler op e; };                      \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                 \
        {                                                                                              \
            const auto lowerBound = i * arrayChunckSize;                                               \
            const auto upperBound = MIN(lowerBound + arrayChunckSize, size);                           \
            const auto nElements = upperBound - lowerBound;                                            \
            if (0 == nElements)                                                                        \
                break;                                                                                 \
            cudaMemcpyAsync(destGPU + lowerBound, dest + lowerBound,                                   \
                            sizeof(output_type) * nElements, cudaMemcpyHostToDevice,                   \
                            streams[i]);                                                               \
            cudaMemcpyAsync(arrGPU + lowerBound, arr + lowerBound,                                     \
                            sizeof(arr_type) * nElements, cudaMemcpyHostToDevice,                      \
                            streams[i]);                                                               \
            const std::size_t threadsPerBlock = nElements > 32 ? 32 : nElements;                       \
            const std::size_t blocksPerGrid = (nElements + threadsPerBlock - 1) / threadsPerBlock;     \
            ArrayMap<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(                               \
                destGPU + lowerBound,                                                                  \
                arrGPU + lowerBound,                                                                   \
                f, nElements);                                                                         \
            cudaMemcpyAsync(dest + lowerBound, destGPU + lowerBound,                                   \
                            sizeof(output_type) * nElements, cudaMemcpyDeviceToHost,                   \
                            streams[i]);                                                               \
        }                                                                                              \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                 \
            cudaStreamSynchronize(streams[i]);                                                         \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                 \
            cudaStreamDestroy(streams[i]);                                                             \
        cudaFree(destGPU);                                                                             \
        cudaFree(arrGPU);                                                                              \
    }

#define SCALER_GPU_ARRAY_FUNCTION_IMPLEMENTATION(output_type, scaler_type, arr_type, func_name, op)    \
    void func_name(output_type *dest, const scaler_type scaler, const arr_type *arr, std::size_t size) \
    {                                                                                                  \
        const auto f = [=] __device__(const arr_type &e) { return scaler op e; };                      \
        const std::size_t threadsPerBlock = size > 32 ? 32 : size;                                     \
        const std::size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;              \
        ArrayMap<<<blocksPerGrid, threadsPerBlock>>>(dest, arr, f, size);                              \
    }

#define SCALER_ARRAY_FUNCTION_IMPLEMENTATIONS(func_name, op) \
    FUNCTION_IMPLEMENTATIONS_WITH_THREE_TYPES(SCALER_ARRAY_FUNCTION_IMPLEMENTATION, func_name, op)

#define SCALER_GPU_ARRAY_FUNCTION_IMPLEMENTATIONS(func_name, op) \
    FUNCTION_IMPLEMENTATIONS_WITH_THREE_TYPES(SCALER_GPU_ARRAY_FUNCTION_IMPLEMENTATION, func_name, op)

namespace CudaHelpers
{
    GPU_ARRAY_POPULATE_FUNCTION_IMPLEMENTATIONS(GPUArrayPopulate, GPUArray::PopulateArray);

    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(AddWithTwoArrays, ArrayAddition);
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(SubtractWithTwoArrays, ArraySubtraction);
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(MultiplyWithTwoArrays, ArrayMultiplication);
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(DivideWithTwoArrays, ArrayDivision);

    TWO_GPU_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(AddWithTwoGPUArrays, ArrayAddition);
    TWO_GPU_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(SubtractWithTwoGPUArrays, ArraySubtraction);
    TWO_GPU_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(MultiplyWithTwoGPUArrays, ArrayMultiplication);
    TWO_GPU_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(DivideWithTwoGPUArrays, ArrayDivision);

    ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(AddWithArrayScaler, ArrayScalerAddition);
    ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(SubtractWithArrayScaler, ArrayScalerSubtraction);
    ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(MultiplyWithArrayScaler, ArrayScalerMultiplication);
    ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(DivideWithArrayScaler, ArrayScalerDivision);

    GPU_ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(AddWithGPUArrayScaler, ArrayScalerAddition);
    GPU_ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(SubtractWithGPUArrayScaler, ArrayScalerSubtraction);
    GPU_ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(MultiplyWithGPUArrayScaler, ArrayScalerMultiplication);
    GPU_ARRAY_SCALER_FUNCTION_IMPLEMENTATIONS(DivideWithGPUArrayScaler, ArrayScalerDivision);

    SCALER_ARRAY_FUNCTION_IMPLEMENTATIONS(AddWithScalerArray, +);
    SCALER_ARRAY_FUNCTION_IMPLEMENTATIONS(SubtractWithScalerArray, -);
    SCALER_ARRAY_FUNCTION_IMPLEMENTATIONS(MultiplyWithScalerArray, *);
    SCALER_ARRAY_FUNCTION_IMPLEMENTATIONS(DivideWithScalerArray, /);

    SCALER_GPU_ARRAY_FUNCTION_IMPLEMENTATIONS(AddWithScalerGPUArray, +);
    SCALER_GPU_ARRAY_FUNCTION_IMPLEMENTATIONS(SubtractWithScalerGPUArray, -);
    SCALER_GPU_ARRAY_FUNCTION_IMPLEMENTATIONS(MultiplyWithScalerGPUArray, *);
    SCALER_GPU_ARRAY_FUNCTION_IMPLEMENTATIONS(DivideWithScalerGPUArray, /);
}