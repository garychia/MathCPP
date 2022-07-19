#ifndef CUDAHELPERS_HPP
#define CUDAHELPERS_HPP

#include <cstdlib>

#define TWO_OPERAND_FUNCTION_SIGNATURES(func_name)                                                  \
    void func_name(int *Dest, const int *Operand1, const int *Operand2, std::size_t size);          \
    void func_name(float *Dest, const int *Operand1, const float *Operand2, std::size_t size);      \
    void func_name(float *Dest, const float *Operand1, const int *Operand2, std::size_t size);      \
    void func_name(float *Dest, const float *Operand1, const float *Operand2, std::size_t size);    \
    void func_name(double *Dest, const int *Operand1, const double *Operand2, std::size_t size);    \
    void func_name(double *Dest, const double *Operand1, const int *Operand2, std::size_t size);    \
    void func_name(double *Dest, const double *Operand1, const double *Operand2, std::size_t size); \
    void func_name(double *Dest, const float *Operand1, const double *Operand2, std::size_t size);  \
    void func_name(double *Dest, const double *Operand1, const float *Operand2, std::size_t size);

#define ARRAY_SCALER_FUNCTION_SIGNATURES(func_name)                                         \
    void func_name(int *dest, const int *arr, const int scaler, std::size_t size);          \
    void func_name(float *dest, const int *arr, const float scaler, std::size_t size);      \
    void func_name(float *dest, const float *arr, const int scaler, std::size_t size);      \
    void func_name(float *dest, const float *arr, const float scaler, std::size_t size);    \
    void func_name(double *dest, const int *arr, const double scaler, std::size_t size);    \
    void func_name(double *dest, const double *arr, const int scaler, std::size_t size);    \
    void func_name(double *dest, const double *arr, const double scaler, std::size_t size); \
    void func_name(double *dest, const float *arr, const double scaler, std::size_t size);  \
    void func_name(double *dest, const double *arr, const float scaler, std::size_t size);

#define SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, output_type, scaler_type, arr_type) \
    void func_name(output_type *dest, const scaler_type scaler, const arr_type *arr, std::size_t size);

#define SCALER_ARRAY_FUNCTION_SIGNATURES(func_name)                    \
    SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, int, int, int)          \
    SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, float, int, float)      \
    SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, float, float, int)      \
    SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, float, float, float)    \
    SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, double, int, double)    \
    SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, double, double, int)    \
    SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, double, double, double) \
    SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, double, float, double)  \
    SCALER_ARRAY_FUNCTION_SIGNATURE(func_name, double, double, float)

namespace CudaHelpers
{
    TWO_OPERAND_FUNCTION_SIGNATURES(AddWithTwoArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(SubtractWithTwoArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(MultiplyWithTwoArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(DivideWithTwoArrays);

    ARRAY_SCALER_FUNCTION_SIGNATURES(AddWithArrayScaler);
    ARRAY_SCALER_FUNCTION_SIGNATURES(SubtractWithArrayScaler);
    ARRAY_SCALER_FUNCTION_SIGNATURES(MultiplyWithArrayScaler);
    ARRAY_SCALER_FUNCTION_SIGNATURES(DivideWithArrayScaler);

    SCALER_ARRAY_FUNCTION_SIGNATURES(AddWithScalerArray);
    SCALER_ARRAY_FUNCTION_SIGNATURES(SubtractWithScalerArray);
    SCALER_ARRAY_FUNCTION_SIGNATURES(MultiplyWithScalerArray);
    SCALER_ARRAY_FUNCTION_SIGNATURES(DivideWithScalerArray);
} // namespace CudaHelpers

#endif