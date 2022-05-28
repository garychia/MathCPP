#include <iostream>
#include <memory>

#include "Math.hpp"

using namespace Math;

int main(void)
{
    std::cout << "sin(0) = " << Sine<float>(0) << std::endl;
    std::cout << "sin(pi / 2) = " << Sine<float>(3.14159265359 / 2) << std::endl;
    std::cout << "sin(pi) = " << Sine<float>(3.14159265359) << std::endl;
    std::cout << "sin(3pi / 2) = " << Sine<float>(3 * 3.14159265359 / 2) << std::endl;
    std::cout << "sin(2pi) = " << Sine<float>(2 * 3.14159265359) << std::endl;
    std::cout << "sin(23) = " << Sine(23.0) << std::endl;
    std::cout << "cos(0) = " << Cosine<float>(0) << std::endl;
    std::cout << "cos(pi / 2) = " << Cosine<float>(3.14159265359 / 2) << std::endl;
    std::cout << "cos(pi) = " << Cosine<float>(3.14159265359) << std::endl;
    std::cout << "cos(3pi / 2) = " << Cosine<float>(3 * 3.14159265359 / 2) << std::endl;
    std::cout << "cos(2pi) = " << Cosine<float>(2 * 3.14159265359) << std::endl;
    std::cout << "cos(23) = " << Cosine<float>(23.0) << std::endl;
    return 0;
}