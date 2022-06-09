#include "NeuralLayer.hpp"
namespace MachineLearning
{
    NeuralLayer::NeuralLayer() : weights(), biases(), output()
    {
    }

    NeuralLayer::NeuralLayer(std::size_t inputSize, std::size_t outputSize) : weights(inputSize, outputSize), biases(outputSize, 1), output()
    {
    }

    NeuralLayer::~NeuralLayer()
    {
    }

    Matrix<double> NeuralLayer::Forward(const Matrix<double> &input)
    {
        return output = weights.Transposed() * input + biases;
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

    Matrix<double> NeuralLayer::GetLastOutput() const
    {
        return output;
    }

    std::ostream &operator<<(std::ostream &stream, const NeuralLayer &layer)
    {
        stream << layer.ToString();
        return stream;
    }
} // namespace MachineLearning
