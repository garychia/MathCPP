#include "ArithmeticOperations.cuh"

#include "CUDAHelpers.hpp"

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
        cudaMemcpy(op1, Operand1, bytesOfArray1, cudaMemcpyHostToDevice);                                         \
        cudaMemcpy(op2, Operand2, bytesOfArray2, cudaMemcpyHostToDevice);                                         \
        const std::size_t threadsPerBlock = 32;                                                                   \
        const std::size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;                         \
        helper_func<<<blocksPerGrid, threadsPerBlock>>>(dest, op1, op2, size);                                    \
        cudaMemcpy(Dest, dest, bytesOfOutput, cudaMemcpyDeviceToHost);                                            \
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