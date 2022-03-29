#include <iostream>

#include "DataStructures/matrix.hpp"

int main(void)
{
    DataStructure::Matrix<double> i1 = DataStructure::Matrix<int>::Identity(10);
    DataStructure::Matrix<int> i2 = DataStructure::Matrix<int>::Identity(10);
    i1 *= 3.14;
    i2 *= 394;
    std::cout << i2 * i1 << std::endl;
    return 0;
}