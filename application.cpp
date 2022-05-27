#include <iostream>
#include <memory>

#include "Matrix.hpp"

using namespace DataStructure;

int main(void)
{
    Vector<float> transV({ 4, 5, 9 });
    Matrix<float> transM = Matrix<float>::Translation(transV);
    std::cout << "Translation Vector:\n" << transV << std::endl;
    std::cout << "Translation Matrix:\n" << transM << std::endl;
    return 0;
}