#include "NeuralLayer.hpp"
namespace MachineLearning
{
    NeuralLayer::NeuralLayer() : weights(), biases()
    {
    }

    NeuralLayer::NeuralLayer(std::size_t inputSize, std::size_t outputSize) : weights(inputSize, outputSize), biases(outputSize, 1)
    {
    }

    NeuralLayer::~NeuralLayer()
    {
    }

    Matrix<double> NeuralLayer::Forward(const Matrix<double> &input) const
    {
        return weights.Transposed() * input + biases;
    }

    std::string NeuralLayer::ToString() const
    {
        std::stringstream ss;
        ss << "NeuralLayer:" << std::endl;
        ss << "Weights:\n"
           << weights << std::endl;
        ss << "Biases:\n"
           << biases;
        return ss.str();
    }
} // namespace MachineLearning
