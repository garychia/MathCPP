#include <iostream>

#include "matrix.hpp"

int main(void)
{
    Math::Tuple<int> v;
    Math::Matrix<double> m1 = {{1.23, 2.25, 3}, {1, 2, 3}};
    std::cout << v << std::endl;
    std::cout << m1 << std::endl;
    return 0;
}