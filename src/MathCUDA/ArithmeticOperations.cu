#include "Math.hpp"

#define ARRAY_ARITHMETIC_FUNCTION(func_name, op)                                                                                           \
	template <class OutputType, class FirstOperandType, class SecondOperandType, class IndexType>                                          \
	__global__ void func_name(OutputType *dest, const FirstOperandType *operand1, const SecondOperandType *operand2, const IndexType size) \
	{                                                                                                                                      \
		const std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;                                                                       \
		if (i < size)                                                                                                                      \
			dest[i] = operand1[i] op operand2[i];                                                                                          \
	}

#define ARRAY_SCALER_FUNCTION(func_name, op)                                                                         \
	template <class OutputType, class ArrayType, class ScalerType, class IndexType>                                  \
	__global__ void func_name(OutputType *dest, const ArrayType *arr, const ScalerType scaler, const IndexType size) \
	{                                                                                                                \
		const std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;                                                 \
		if (i < size)                                                                                                \
			dest[i] = arr[i] op scaler;                                                                              \
	}

ARRAY_ARITHMETIC_FUNCTION(ArrayAddition, +);
ARRAY_ARITHMETIC_FUNCTION(ArraySubtraction, -);
ARRAY_ARITHMETIC_FUNCTION(ArrayMultiplication, *);
ARRAY_ARITHMETIC_FUNCTION(ArrayDivision, /);

ARRAY_SCALER_FUNCTION(ArrayScalerAddition, +);
ARRAY_SCALER_FUNCTION(ArrayScalerSubtraction, -);
ARRAY_SCALER_FUNCTION(ArrayScalerMultiplication, *);
ARRAY_SCALER_FUNCTION(ArrayScalerDivision, /);

template <class T, class U, class MapFunction>
__global__ void ArrayMap(T *output, const U *input, MapFunction f, std::size_t size)
{
	const std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size)
		output[i] = f(input[i]);
}

template <class T, class IndexType, class PowerType>
__global__ void Power(T *dest, const T *arr, IndexType size, PowerType power)
{
	const IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size)
		dest[i] = Math::Power(arr[i], power);
}