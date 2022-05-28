#include <iostream>
#include <memory>

#include "Matrix.hpp"

using namespace DataStructure;

int main(void)
{
    Vector<double> axis({23.23, 5.3, -8.8});
    double angle = 7.43;
    auto rotation = Matrix<double>::Rotation3D(axis, angle);
    std::cout << "Rotate around " << axis << " by " << angle << " radians.\n";
    std::cout << "Rotation Matrix:" << std::endl;
    std::cout << rotation << std::endl;
    return 0;
}