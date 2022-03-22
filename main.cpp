#include <iostream>

#include "vector.hpp"

int main(void) {
    Math::Vector3D<int> v1(32, 32, 43);
    Math::Vector3D<double> v2(23, 23.32, 23.23);
    auto v3 = v1.Add(v2);
    std::cout << v3.Z << std::endl;
    return 0;
}