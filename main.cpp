#include <iostream>

#include "matrix.hpp"

int main(void)
{
    DataStructure::Matrix<int> i1 = DataStructure::Matrix<int>::Identity(10);
    DataStructure::Matrix<int> i2 = DataStructure::Matrix<int>::Identity(10);
    std::cout << i1.Multiply(i2) << std::endl;
    return 0;
}