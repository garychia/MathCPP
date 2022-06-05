#include <iostream>

#include "MLAlgs.hpp"

/*
class NeuralLayer
{
private:
    Matrix<double> weights;
public:
    NeuralLayer(std::size_t inputSize);
    ~NeuralLayer();
};

NeuralLayer::NeuralLayer()
{
}

NeuralLayer::~NeuralLayer()
{
}


class NeuralNetwork
{
private:
    List<NeuralLayer> layers;
public:
    NeuralNetwork();
    ~NeuralNetwork();
};

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::~NeuralNetwork()
{
}
*/

using namespace DataStructure;

int main(void)
{
    Matrix<float> input({1, 2, 3});
    auto prediction = -1.f;
    auto label = 1.f;
    std::cout << "Input =\n" << input << std::endl;
    std::cout << "Prediction = " << prediction << std::endl;
    std::cout << "Label = " << label << std::endl;
    std::cout << "Hinge Loss:\n" << MLAlgs::HingeLoss(prediction, label) << std::endl;
    std::cout << "Hinge Loss Gradient:\n" << MLAlgs::HingeLossGradient(input, prediction, label) << std::endl;
    return 0;
}