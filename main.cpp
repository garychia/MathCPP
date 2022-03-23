#include <iostream>

#include "vector3d.hpp"
#include "vector.hpp"

int main(void)
{
    Math::Vector3D<int> v1(32, 32, 43);
    Math::Vector3D<double> v2(23, 23.32, 23.23);
    Math::Vector3D<float> v3(3.14f, 234.234f, 324.f);
    Math::Vector<int> v({1, 2, 3});

    std::cout << v1 << std::endl;
    std::cout << v2 << std::endl;
    std::cout << v3 << std::endl;

    for (int i = 0; i < 3; i++)
        std::cout << v1[i] << std::endl;

    try
    {
        std::cout << v1[-3] << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    for (int i = 0; i < 3; i++)
        std::cout << v[i] << std::endl;

    try
    {
        std::cout << v[-12] << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}