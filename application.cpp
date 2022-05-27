#include <iostream>
#include <memory>

#include "Matrix.hpp"

using namespace DataStructure;

int main(void)
{
    Vector<float> factors({ 4, 5, 9 });
    Matrix<float> scaleM = Matrix<float>::Scaling(factors);
    std::cout << "Scaling Vector:\n" << factors << std::endl;
    std::cout << "Scaling Matrix:\n" << scaleM << std::endl;
    return 0;
}