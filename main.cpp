#include <iostream>

#include "matrix.hpp"

int main(void)
{
    DataStructure::Vector<int> v1 = {1, 2, 3};
    DataStructure::Vector<int> v2 = {2, 3, 4};
    std::cout << v1 << std::endl;
    std::cout << v2 << std::endl;
    std::cout << v1 * 3.14 << std::endl;
    return 0;
}