#include "LinearLayer.hpp"

#include <sstream>

namespace MachineLearning
{
    LinearLayer::LinearLayer() : NeuralLayer(), weights(), biases(), dWeights(), dBiases()
    {
    }

    LinearLayer::LinearLayer(std::size_t inputSize, std::size_t outputSize) : NeuralLayer(), weights(inputSize, outputSize), biases(outputSize, 1), dWeights(), dBiases()
    {
    }

    Matrix<double> LinearLayer::Forward(const Matrix<double> &input)
    {
        this->input = input;
        return this->output = weights.Transposed() * input + biases;
    }

    Matrix<double> LinearLayer::Backward(const Matrix<double> &derivative)
    {
        dWeights = this->input * derivative.Transposed();
        dBiases = derivative.Sum(false);
        return this->weights * derivative;
    }

    void LinearLayer::UpdateWeights(const double &learningRate)
    {
        weights -= learningRate * dWeights;
        dBiases -= learningRate * dBiases;
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
