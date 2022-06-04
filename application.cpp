#include <iostream>
#include "Math.hpp"

using namespace DataStructure;

int main(void)
{
    Matrix<float> m({
        {1, 4, 6, 3},
        {1, 3, 5, 0},
        {1, 1, 1, 1},
        {1, 2, 3, 4}
        });
    m.Transpose();
    std::cout << "M = \n" << m << std::endl;
    std::cout << "softmax(M) = \n" << Math::Softmax(m) << std::endl;
    return 0;
}