#include <iostream>
#include <memory>

#include "Matrix.hpp"

using namespace DataStructure;

int main(void)
{
    Matrix<int> a(
        {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        }
    );
    std::cout << "Matrix a =\n";
    std::cout << a << std::endl;
    std::cout << "Flattened in row-major order as a single row:\n";
    std::cout << a.Flattened() << std::endl;
    std::cout << "Flattened in row-major order as a single column:\n";
    std::cout << a.Flattened(true, false) << std::endl;
    std::cout << "Flattened in column-major order as a single row:\n";
    std::cout << a.Flattened(false) << std::endl;
    std::cout << "Flattened in column-major order as a single column:\n";
    std::cout << a.Flattened(false, false) << std::endl;
    return 0;
}