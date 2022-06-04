#include <iostream>
#include "Math.hpp"

using namespace DataStructure;

#define PRINT_MATH_FUNCTION_RESULT(input, math_func) \
    std::cout << #input " = " << input << std::endl; \
    std::cout << #math_func "(" #input ") = " << math_func(input) << std::endl;

int main(void)
{
    List<double> values({-8, -1, 0, 1, 8});
    for (std::size_t i = 0; i < values.Size(); i++)
    {
        double x = values[i];
        PRINT_MATH_FUNCTION_RESULT(x, Math::Sinh);
        PRINT_MATH_FUNCTION_RESULT(x, Math::Cosh);
        PRINT_MATH_FUNCTION_RESULT(x, Math::Tanh);
        PRINT_MATH_FUNCTION_RESULT(x, Math::Exponent)
    }
    return 0;
}