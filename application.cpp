#include <iostream>

#include "MachineLearning/NeuralNetwork.hpp"

using namespace DataStructures;

int main(void)
{
    Matrix<double> input({1, 2, 3});
    MachineLearning::NeuralNetwork network;
    network.AddLayer(MachineLearning::NeuralLayer(3, 1));
    std::cout << "Input =\n"
              << input << std::endl;
    try
    {
        std::cout << "Prediction =\n"
              << network.Predict(input) << std::endl;
    }
    catch (const Exceptions::MatrixShapeMismatch &e)
    {
        std::cout << e.what() << std::endl;
    }
    return 0;
}