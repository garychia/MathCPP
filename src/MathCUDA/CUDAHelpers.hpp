#ifndef CUDAHELPERS_HPP
#define CUDAHELPERS_HPP

#include <cstdlib>

#define EXPAND(x) x

#define TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(macro, type1, type2, ...) \
    EXPAND(macro(type1, type1, type2, __VA_ARGS__));                   \
    EXPAND(macro(type1, type2, type1, __VA_ARGS__));                   \
    EXPAND(macro(type2, type1, type1, __VA_ARGS__));                   \
    EXPAND(macro(type2, type2, type1, __VA_ARGS__));                   \
    EXPAND(macro(type2, type1, type2, __VA_ARGS__));                   \
    EXPAND(macro(type1, type2, type2, __VA_ARGS__));

#define THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(macro, type1, type2, type3, ...) \
    EXPAND(macro(type1, type2, type3, __VA_ARGS__));                            \
    EXPAND(macro(type1, type3, type2, __VA_ARGS__));                            \
    EXPAND(macro(type2, type1, type3, __VA_ARGS__));                            \
    EXPAND(macro(type2, type3, type1, __VA_ARGS__));                            \
    EXPAND(macro(type3, type1, type2, __VA_ARGS__));                            \
    EXPAND(macro(type3, type2, type1, __VA_ARGS__));

#define TWO_OPERAND_FUNCTION_SIGNATURE(type1, type2, type3, func_name) \
    void func_name(type1 *dest, const type2 *op1, const type3 *op2, std::size_t size)

#define TWO_OPERAND_FUNCTION_SIGNATURES(func_name)                                                 \
    TWO_OPERAND_FUNCTION_SIGNATURE(int, int, int, func_name);                                      \
    TWO_OPERAND_FUNCTION_SIGNATURE(float, float, float, func_name);                                \
    TWO_OPERAND_FUNCTION_SIGNATURE(double, double, double, func_name);                             \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(TWO_OPERAND_FUNCTION_SIGNATURE, int, float, func_name)    \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(TWO_OPERAND_FUNCTION_SIGNATURE, int, double, func_name)   \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(TWO_OPERAND_FUNCTION_SIGNATURE, float, double, func_name) \
    THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(TWO_OPERAND_FUNCTION_SIGNATURE, int, float, double, func_name)

#define GPU_ARRAY_POPULATE_FUNCTION_SIGNATURES(func_name)       \
    void func_name(int *dest, std::size_t size, int value);     \
    void func_name(float *dest, std::size_t size, float value); \
    void func_name(double *dest, std::size_t size, double value);

#define ARRAY_SCALER_FUNCTION_SIGNATURE(output_type, arr_type, scaler_type, func_name) \
    void func_name(output_type *dest, const arr_type *arr, const scaler_type scaler, std::size_t size)

#define ARRAY_SCALER_FUNCTION_SIGNATURES(func_name)                                                                \
    ARRAY_SCALER_FUNCTION_SIGNATURE(int, int, int, func_name);                                                     \
    ARRAY_SCALER_FUNCTION_SIGNATURE(float, float, float, func_name);                                               \
    ARRAY_SCALER_FUNCTION_SIGNATURE(double, double, double, func_name);                                            \
    ARRAY_SCALER_FUNCTION_SIGNATURE(std::size_t, std::size_t, std::size_t, func_name);                             \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, int, float, func_name)                   \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, int, double, func_name)                  \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, int, std::size_t, func_name)             \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, float, double, func_name)                \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, float, std::size_t, func_name)           \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, double, std::size_t, func_name)          \
    THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, int, float, std::size_t, func_name)    \
    THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, int, std::size_t, double, func_name)   \
    THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, std::size_t, float, double, func_name) \
    THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(ARRAY_SCALER_FUNCTION_SIGNATURE, int, float, double, func_name)

#define SCALER_ARRAY_FUNCTION_SIGNATURE(output_type, scaler_type, arr_type, func_name) \
    void func_name(output_type *dest, const scaler_type scaler, const arr_type *arr, std::size_t size)

#define SCALER_ARRAY_FUNCTION_SIGNATURES(func_name)                                                                \
    SCALER_ARRAY_FUNCTION_SIGNATURE(int, int, int, func_name);                                                     \
    SCALER_ARRAY_FUNCTION_SIGNATURE(float, float, float, func_name);                                               \
    SCALER_ARRAY_FUNCTION_SIGNATURE(double, double, double, func_name);                                            \
    SCALER_ARRAY_FUNCTION_SIGNATURE(std::size_t, std::size_t, std::size_t, func_name);                             \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, int, float, func_name)                   \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, int, double, func_name)                  \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, int, std::size_t, func_name)             \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, float, double, func_name)                \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, float, std::size_t, func_name)           \
    TWO_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, double, std::size_t, func_name)          \
    THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, int, float, std::size_t, func_name)    \
    THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, int, std::size_t, double, func_name)   \
    THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, std::size_t, float, double, func_name) \
    THREE_TYPE_PERMUTATION_WITH_THREE_ARGS(SCALER_ARRAY_FUNCTION_SIGNATURE, int, float, double, func_name)

namespace CudaHelpers
{
    GPU_ARRAY_POPULATE_FUNCTION_SIGNATURES(GPUArrayPopulate);

    TWO_OPERAND_FUNCTION_SIGNATURES(AddWithTwoArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(SubtractWithTwoArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(MultiplyWithTwoArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(DivideWithTwoArrays);

    TWO_OPERAND_FUNCTION_SIGNATURES(AddWithTwoGPUArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(SubtractWithTwoGPUArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(MultiplyWithTwoGPUArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(DivideWithTwoGPUArrays);

    ARRAY_SCALER_FUNCTION_SIGNATURES(AddWithArrayScaler);
    ARRAY_SCALER_FUNCTION_SIGNATURES(SubtractWithArrayScaler);
    ARRAY_SCALER_FUNCTION_SIGNATURES(MultiplyWithArrayScaler);
    ARRAY_SCALER_FUNCTION_SIGNATURES(DivideWithArrayScaler);

    ARRAY_SCALER_FUNCTION_SIGNATURES(AddWithGPUArrayScaler);
    ARRAY_SCALER_FUNCTION_SIGNATURES(SubtractWithGPUArrayScaler);
    ARRAY_SCALER_FUNCTION_SIGNATURES(MultiplyWithGPUArrayScaler);
    ARRAY_SCALER_FUNCTION_SIGNATURES(DivideWithGPUArrayScaler);

    SCALER_ARRAY_FUNCTION_SIGNATURES(AddWithScalerArray);
    SCALER_ARRAY_FUNCTION_SIGNATURES(SubtractWithScalerArray);
    SCALER_ARRAY_FUNCTION_SIGNATURES(MultiplyWithScalerArray);
    SCALER_ARRAY_FUNCTION_SIGNATURES(DivideWithScalerArray);

    SCALER_ARRAY_FUNCTION_SIGNATURES(AddWithScalerGPUArray);
    SCALER_ARRAY_FUNCTION_SIGNATURES(SubtractWithScalerGPUArray);
    SCALER_ARRAY_FUNCTION_SIGNATURES(MultiplyWithScalerGPUArray);
    SCALER_ARRAY_FUNCTION_SIGNATURES(DivideWithScalerGPUArray);
} // namespace CudaHelpers

#endif