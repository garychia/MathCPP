#include "Math.hpp"

#define ARITHMETIC_FUNCTION(func_name, op)                                                                                                  \
	template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>                                           \
	__global__ void func_name(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2, const IndexType size) \
	{                                                                                                                                       \
		const std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;                                                                        \
		if (i < size)                                                                                                                       \
			dest[i] = operand1[i] op operand2[i];                                                                                           \
	}

ARITHMETIC_FUNCTION(ArrayAddition, +);
ARITHMETIC_FUNCTION(ArraySubtraction, -);
ARITHMETIC_FUNCTION(ArrayMultiplication, *);
ARITHMETIC_FUNCTION(ArrayDivision, /);

template <class T, class IndexType, class PowerType>
__global__ void Power(T *dest, const T *arr, IndexType size, PowerType power)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size)
		dest[i] = Math::Power(arr[i], power);
}