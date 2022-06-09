#include "NeuralNetwork.hpp"

namespace MachineLearning
{
    NeuralNetwork::NeuralNetwork(const List<NeuralLayer> &layers) : layers(layers)
    {
    }

    void NeuralNetwork::AddLayer(const NeuralLayer &layer)
    {
        layers.Append(layer);
    }

    Matrix<double> NeuralNetwork::Predict(const Matrix<double> &input)
    {
        Matrix<double> output = input;
        for (std::size_t i = 0; i < layers.Size(); i++)
            output = layers[i].Forward(output);
        return output;
    }
} // namespace MachineLearning
