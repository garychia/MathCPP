#include <iostream>
#include "Math.hpp"

using namespace DataStructure;

int main(void)
{
    Vector<float> v({1, 4, 6, 3});
    std::cout << "v = " << v << std::endl;
    std::cout << "softmax(v) = " << Math::Softmax(v) << std::endl;
    return 0;
}