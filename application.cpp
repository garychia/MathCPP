#include <iostream>

#include "Algorithms/math.hpp"
#include "DataStructures/matrix.hpp"

using namespace DataStructure;

int main(void)
{
    auto diagonalM = Matrix<int>::Diagonal(Vector<int>({ 2, 5, 23, 89, 23 }));
    std::cout << Math::Power(diagonalM, 3) << std::endl;
    return 0;
}