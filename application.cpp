#include <iostream>

#include "MachineLearning/NeuralNetwork.hpp"

#include <stdlib.h>

using namespace DataStructures;
using namespace MachineLearning;

int main(void)
{
    Matrix<double> input({{2, 3, 9, 12}, {5, 2, 6, 5}});
    Matrix<double> labels({{0, 1, 0, 1}, {1, 0, 1, 0}});
    List<NeuralNetwork::LayerType> layerTypes({NeuralNetwork::LayerType::Linear, NeuralNetwork::LayerType::SoftMax});
    List<Tuple<unsigned int>> shapes({{2, 2}, {2, 2}});
    NeuralNetwork network(layerTypes, shapes, NeuralNetwork::LossType::NLL);
    std::cout << network << std::endl;
    std::cout << network.Predict(input) << std::endl;
    const auto history = network.Train(input, labels, 1, 0.01);
    std::cout << history << std::endl;
    return 0;
}