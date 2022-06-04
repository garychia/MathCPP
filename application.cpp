#include <iostream>

#include "Math.hpp"

using namespace DataStructure;

int main(void)
{
    Vector<float> values({1, 1, 2, 5, 6, 6, 6, 6, 5, 1, 0, 100});
    const float mean = values.Sum() / values.Size();
    const float sigma =
        Math::Power(
            values
                .Map([&mean](float e)
                     { return Math::Power(e - mean, 2); })
                .Sum(),
            0.5);
    std::cout << "Sequence: " << values << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Standard Deviation: " << sigma << std::endl;
    std::cout << "Likelyhood:" << std::endl;
    for (int i = 0; i < values.Size(); i++)
        std::cout << Math::Gauss(values[i], mean, sigma) << " at " << values[i] << std::endl;
    std::cout << Math::Gauss(mean, mean, sigma) << " at " << mean << std::endl;
    return 0;
}