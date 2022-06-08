#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "NeuralLayer.hpp"

#include <string>
#include <sstream>

namespace MachineLearning
{
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
} // namespace MachineLearning

#endif // NEURALNETWORK_HPP