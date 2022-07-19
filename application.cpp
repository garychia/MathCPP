#include <iostream>
#include <stdlib.h>

#include "Matrix.hpp"

using namespace DataStructures;

int main(void)
{
    Matrix<double> m(
        {
            {1, 2, 1},
            {3, 8, 1},
            {0, 4, 1}
        });
    std::cout << "Matrix Before Elimination:\n" << m << std::endl;
    m.Eliminate();
    std::cout << "Matrix After Elimination:\n" << m << std::endl;
    return 0;
}