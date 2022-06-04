#include <iostream>
#include "Math.hpp"

using namespace DataStructure;

int main(void)
{
    float x = -1;
    std::cout << "x = " << x << "\n";
    std::cout << "sigmoid(x) = " << Math::Sigmoid(x) << "\n";
    x = 0;
    std::cout << "x = " << x << "\n";
    std::cout << "sigmoid(x) = " << Math::Sigmoid(x) << "\n";
    x = 1;
    std::cout << "x = " << x << "\n";
    std::cout << "sigmoid(x) = " << Math::Sigmoid(x) << "\n";
    return 0;
}