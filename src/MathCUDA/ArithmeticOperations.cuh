#ifndef ARITHMETICOPERATIONS_CUH
#define ARITHMETICOPERATIONS_CUH

#include "cuda_runtime.h"

/**
 * @brief Perform element-wise addition on two array.
 * 
 * @tparam T the type of the array elements.
 * @tparam IndexType the type of the index.
 * @param dest the destination array where the results will be stored.
 * @param operand1 the first array as an input to the addition.
 * @param operand2 the second array as an input to the addition.
 * @param size the total number of pairs of array elements to be added.
 */
template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>
__global__ void ArrayAddition(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2, const IndexType size);

template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>
__global__ void ArraySubtraction(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2, const IndexType size);

template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>
__global__ void ArrayMultiplication(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2, const IndexType size);

template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>
__global__ void ArrayDivision(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2, const IndexType size);

/**
 * @brief Calculate each element of an array raised to a given power.
 * 
 * @tparam T the type of the array element.
 * @tparam IndexType the type of the index.
 * @tparam PowerType the type of the power.
 * @param dest the destination array where the results will be stored.
 * @param arr an array.
 * @param size the number of elements in the array.
 * @param power the power.
 */
template <class T, class IndexType, class PowerType>
__global__ void Power(T *dest, const T *arr, IndexType size, PowerType power);

#include "ArithmeticOperations.cu"

#endif