#include <iostream>

#include "MachineLearning/NeuralNetwork.hpp"
#include "MachineLearning/LinearLayer.hpp"

using namespace DataStructures;

int main(void)
{
    Matrix<double> input({1, 2, 3});
    MachineLearning::NeuralNetwork network;
    auto linearLayer = new MachineLearning::LinearLayer(3, 1);
    network.AddLayer(linearLayer);
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
    delete linearLayer;
    return 0;
}