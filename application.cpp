#include <iostream>
#include <memory>

#include "Matrix.hpp"

using namespace DataStructure;

int main(void)
{
    Vector<float> vec({1, 2, 3, 4, 5, 100});
    const float *vecPtr = vec.AsRawPointer();
    std::cout << "vec = " << vec << std::endl;
    std::cout << "Elements in the array are:" << std::endl;
    for (std::size_t i = 0; i < vec.Size(); i++)
        std::cout << vecPtr[i] << std::endl;
    return 0;
}