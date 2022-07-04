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
} // namespace CudaHelpers

#endif