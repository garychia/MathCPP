#include <iostream>

#include "MLAlgs.hpp"

using namespace DataStructure;

int main(void)
{
    Matrix<float> input({1, 2, 3});
    auto prediction = 0.5f;
    auto label = 1.f;
    std::cout << "Input =\n" << input << std::endl;
    std::cout << "Prediction = " << prediction << std::endl;
    std::cout << "Label = " << label << std::endl;
    std::cout << "Hinge Loss:\n" << MLAlgs::HingeLoss(prediction, label) << std::endl;
    std::cout << "Hinge Loss Gradient:\n" << MLAlgs::HingeLossGradient(input, prediction, label) << std::endl;
    return 0;
}