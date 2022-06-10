#include <iostream>

#include "MachineLearning/NeuralNetwork.hpp"

#include <stdlib.h>

using namespace DataStructures;
using namespace MachineLearning;

int main(void)
{
    Matrix<double> input({{2, 3, 9, 12}, {5, 2, 6, 5}});
    Matrix<double> labels({{0, 1, 0, 1}, {1, 0, 1, 0}});
    List<NeuralNetwork::LayerType> layerTypes({NeuralNetwork::LayerType::Linear});
    List<Tuple<unsigned int>> shapes({{2, 3}});
    NeuralNetwork network(layerTypes, shapes);
    
    std::cout << "Input =\n" << input << std::endl;
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