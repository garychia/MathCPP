#include "ArithmeticOperations.cuh"

#include "CUDAHelpers.hpp"

#define MIN(a, b) (a) < (b) ? (a) : (b)

#define TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, output_type, op_type_1, op_type_2, helper_func) \
    void func_name(output_type *Dest, const op_type_1 *Operand1, const op_type_2 *Operand2, std::size_t size)     \
    {                                                                                                             \
        const std::size_t bytesOfOutput = size * sizeof(output_type);                                             \
        const std::size_t bytesOfArray1 = size * sizeof(op_type_1);                                               \
        const std::size_t bytesOfArray2 = size * sizeof(op_type_2);                                               \
        output_type *dest;                                                                                        \
        cudaMalloc(&dest, bytesOfOutput);                                                                         \
        op_type_1 *op1;                                                                                           \
        cudaMalloc(&op1, bytesOfArray1);                                                                          \
        op_type_2 *op2;                                                                                           \
        cudaMalloc(&op2, bytesOfArray2);                                                                          \
        const std::size_t numOfStreams = 8;                                                                       \
        const std::size_t arrayChunckSize = (size + numOfStreams - 1) / numOfStreams;                             \
        cudaStream_t streams[numOfStreams];                                                                       \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                            \
            cudaStreamCreate(&streams[i]);                                                                        \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                            \
        {                                                                                                         \
            const auto lowerBound = i * arrayChunckSize;                                                          \
            const auto upperBound = MIN(lowerBound + arrayChunckSize, size);                                      \
            const auto nElements = upperBound - lowerBound;                                                       \
            cudaMemcpyAsync(dest + lowerBound, Dest + lowerBound,                                                 \
                            sizeof(output_type) * nElements, cudaMemcpyHostToDevice,                              \
                            streams[i]);                                                                          \
            cudaMemcpyAsync(op1 + lowerBound, Operand1 + lowerBound,                                              \
                            sizeof(op_type_1) * nElements, cudaMemcpyHostToDevice,                                \
                            streams[i]);                                                                          \
            cudaMemcpyAsync(op2 + lowerBound, Operand2 + lowerBound,                                              \
                            sizeof(op_type_2) * nElements, cudaMemcpyHostToDevice,                                \
                            streams[i]);                                                                          \
            const std::size_t threadsPerBlock = nElements > 32 ? 32 : (nElements > 0 ? nElements : 1);            \
            const std::size_t blocksPerGrid = (nElements + threadsPerBlock - 1) / threadsPerBlock;                \
            helper_func<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(                                       \
                dest + lowerBound,                                                                                \
                op1 + lowerBound,                                                                                 \
                op2 + lowerBound,                                                                                 \
                nElements);                                                                                       \
            cudaMemcpyAsync(Dest + lowerBound, dest + lowerBound,                                                 \
                            sizeof(output_type) * nElements, cudaMemcpyDeviceToHost,                              \
                            streams[i]);                                                                          \
        }                                                                                                         \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                            \
            cudaStreamSynchronize(streams[i]);                                                                    \
        for (std::size_t i = 0; i < numOfStreams; i++)                                                            \
            cudaStreamDestroy(streams[i]);                                                                        \
        cudaFree(dest);                                                                                           \
        cudaFree(op1);                                                                                            \
        cudaFree(op2);                                                                                            \
    }

#define TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(func_name, helper_func)                    \
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, int, int, int, helper_func)          \
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, float, int, float, helper_func)      \
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, float, float, int, helper_func)      \
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, float, float, float, helper_func)    \
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, double, int, double, helper_func)    \
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, double, double, int, helper_func)    \
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, double, double, double, helper_func) \
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, double, float, double, helper_func)  \
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATION(func_name, double, double, float, helper_func)

namespace CudaHelpers
{
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(AddWithTwoArrays, ArrayAddition)
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(SubtractWithTwoArrays, ArraySubtraction)
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(MultiplyWithTwoArrays, ArrayMultiplication)
    TWO_OPERAND_ARITHMETIC_FUNCTION_IMPLEMENTATIONS(DivideWithTwoArrays, ArrayDivision)
}