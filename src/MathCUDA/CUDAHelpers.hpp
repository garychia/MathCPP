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

namespace CudaHelpers
{
    TWO_OPERAND_FUNCTION_SIGNATURES(AddWithTwoArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(SubtractWithTwoArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(MultiplyWithTwoArrays);
    TWO_OPERAND_FUNCTION_SIGNATURES(DivideWithTwoArrays);
} // namespace CudaHelpers

#endif