#include <iostream>

#include "matrix.hpp"

int main(void)
{
    DataStructure::Vector<int> v1 = {1, 2, 3};
    DataStructure::Vector<int> v2 = {2, 3, 4};
    std::cout << v1 << std::endl;
    std::cout << v2 << std::endl;
    std::cout << v1.Divide(3.0) << std::endl;
    return 0;
}