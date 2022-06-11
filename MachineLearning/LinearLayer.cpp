#include "LinearLayer.hpp"
#include "Math.hpp"
#include "Random.hpp"

#include <sstream>

namespace MachineLearning
{
    LinearLayer::LinearLayer() : NeuralLayer(), weights(), biases(), dWeights(), dBiases()
    {
    }

    LinearLayer::LinearLayer(std::size_t inputSize, std::size_t outputSize) : NeuralLayer(), weights(inputSize, outputSize), biases(outputSize, 1), dWeights(), dBiases()
    {
        const auto sigma = Math::Power<double, double>(inputSize, -0.5);
        weights = weights.Map(
            [&sigma](const double &)
            { return Random::NormalDistribution(0, sigma); });
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
        ss << "LinearLayer: {" << std::endl;
        ss << "  Weights: {\n";
        ss << "    ";
        for (const auto &c : weights.ToString())
        {
            ss << c;
            if (c == '\n')
                ss << "    ";
        }
        ss << "\n  },\n";
        ss << "  Biases: {\n";
        ss << "    ";
        for (const auto &c : biases.ToString())
        {
            ss << c;
            if (c == '\n')
                ss << "    ";
        }
        ss << "\n  }\n";
        ss << "}";
        return ss.str();
    }
} // namespace MachineLearning
