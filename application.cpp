#include <iostream>
#include <memory>

#include "Math.hpp"

using namespace Math;

int main(void)
{
    std::cout << "e^23 = " << Exponent(23.0) << std::endl;
    std::cout << "ln(e^23) = " << NaturalLog<double>(Exponent<double>(23)) << std::endl;
    return 0;
}