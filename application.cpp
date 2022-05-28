#include <iostream>
#include <memory>

#include "Matrix.hpp"

using namespace DataStructure;

int main(void)
{
    float angle = PI / 2;
    auto rotation = Matrix<float>::Rotation2D(angle);
    Matrix<float> point({{32.3}, {64.3}, {1}});
    std::cout << "Rotating ("
              << point[0][0]
              << ", "
              << point[1][0]
              << ") by "
              << angle
              << " radians."
              << std::endl;
    std::cout << "Rotation Matrix:" << std::endl;
    std::cout << rotation << std::endl;
    auto pointAfterRotate = rotation * point;
    std::cout << "The point after the rotation is ("
              << pointAfterRotate[0][0]
              << ", "
              << pointAfterRotate[1][0]
              << ").\n";
    return 0;
}