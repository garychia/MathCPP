#include "NeuralNetwork.hpp"

namespace MachineLearning
{
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
