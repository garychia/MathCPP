#include <iostream>

#include "DataStructures/matrix.hpp"

using namespace DataStructure;

int main(void)
{
    auto diagonalM = Matrix<int>::Diagonal(Vector<int>({ 2, 5, 23, 89, 23 }));
    std::cout << diagonalM << std::endl;
    return 0;
}