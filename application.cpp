#include <iostream>

#include "Math.hpp"

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
    for (int i = 0; i < 10; i++)
        std::cout << Math::Power(2, i) << std::endl;
    return 0;
}