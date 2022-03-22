#include <iostream>

#include "vector3d.hpp"

int main(void) {
    Math::Vector3D<int> v1(32, 32, 43);
    Math::Vector3D<double> v2(23, 23.32, 23.23);
    Math::Vector3D<float> v3(3.14f, 234.234f, 324.f);

    std::cout << v1 << std::endl;
    std::cout << v2 << std::endl;
    std::cout << v3 << std::endl;
    return 0;
}