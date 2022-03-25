#include <iostream>

#include "matrix.hpp"

int main(void)
{
    DataStructure::Matrix<double> m1 = {{1.23, 2.25, 3}, {1, 2, 3}};
    std::cout << "Matrix before Tranpose:" << std::endl;
    std::cout << m1.Shape() << std::endl;
    std::cout << m1 << std::endl;
    std::cout << "After tranpose:" << std::endl;
    m1.Transpose();
    std::cout << m1.Shape() << std::endl;
    std::cout << m1 << std::endl;
    return 0;
}