#include <iostream>

#include "Algorithms/math.hpp"
#include "DataStructures/matrix.hpp"

using namespace DataStructure;

int main(void)
{
    auto diagonalM = Matrix<int>::Diagonal(Vector<int>({ 1, 5, 10, 10, 1 }));
    std::cout << Math::FrobeniusNorm(diagonalM) << std::endl;
    return 0;
}