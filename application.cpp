#include <iostream>
#include <memory>

#include "Matrix.hpp"

using namespace DataStructure;

int main(void)
{
    Matrix<int> a(
        {
            {3, 45, 63, 434},
            {23, 43, 3, 12},
            {3, 4, 32, 13}
        }
    );
    std::cout << "Matrix a =\n";
    std::cout << a << std::endl;
    std::cout << "Flattened in row-major order:\n";
    std::cout << a.Flattened() << std::endl;
    std::cout << "Flattened in column-major order:\n";
    std::cout << a.Flattened(false) << std::endl;
    return 0;
}