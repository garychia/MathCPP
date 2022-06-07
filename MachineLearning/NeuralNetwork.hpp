#include "Matrix.hpp"

#include <string>
#include <sstream>

namespace MachineLearning
{
    class NeuralLayer
    {
    private:
        Matrix<double> weights;
        Matrix<double> biases;

    public:
        NeuralLayer();
        NeuralLayer(std::size_t inputSize, std::size_t outputSize);
        Matrix<double> Forward(const Matrix<double> &input) const;
        ~NeuralLayer();
        std::string ToString() const;

        friend std::ostream &operator<<(std::ostream &stream, const NeuralLayer &layer)
        {
            stream << layer.ToString();
            return stream;
        }
    };

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

    class NeuralNetwork
    {
    private:
        List<NeuralLayer> layers;

    public:
        NeuralNetwork(const List<NeuralLayer> &layers = List<NeuralLayer>());
        ~NeuralNetwork();
        void AddLayer(const NeuralLayer &layer);
        Matrix<double> Predict(const Matrix<double> &input) const;
    };

    NeuralNetwork::NeuralNetwork(const List<NeuralLayer> &layers) : layers(layers)
    {
    }

    NeuralNetwork::~NeuralNetwork()
    {
    }

    void NeuralNetwork::AddLayer(const NeuralLayer &layer)
    {
        layers.Append(layer);
    }

    Matrix<double> NeuralNetwork::Predict(const Matrix<double> &input) const
    {
        Matrix<double> output = input;
        for (std::size_t i = 0; i < layers.Size(); i++)
            output = layers[i].Forward(output);
        return output;
    }
} // namespace MachineLearning