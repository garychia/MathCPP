#include "LinearLayer.hpp"

#include <sstream>

namespace MachineLearning
{
    LinearLayer::LinearLayer() : NeuralLayer(), weights(), biases() 
    {
    }

    LinearLayer::LinearLayer(std::size_t inputSize, std::size_t outputSize) : NeuralLayer(), weights(inputSize, outputSize), biases(outputSize, 1)
    {
    }

    Matrix<double> LinearLayer::Forward(const Matrix<double> &input)
    {
        return this->output = weights.Transposed() * input + biases;
    }

    std::string LinearLayer::ToString() const
    {
        std::stringstream ss;
        ss << "LinearLayer:" << std::endl;
        ss << "Weights:\n"
           << weights << std::endl;
        ss << "Biases:\n"
           << biases;
        return ss.str();
    }
} // namespace MachineLearning
