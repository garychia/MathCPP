#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "NeuralLayer.hpp"

#include <string>
#include <sstream>

namespace MachineLearning
{
    // Neural Network
    class NeuralNetwork
    {
    private:
        // Layers
        List<NeuralLayer *> layers;

    public:
        /**
         * NeuralNetwork Constructor
         * @param layers the layers this neural network will have.
         **/
        NeuralNetwork(const List<NeuralLayer *> &layers = List<NeuralLayer *>());
        // NeuralNetwork Destructor
        ~NeuralNetwork() = default;
        /**
         * Add a neural network layer to this network.
         * @param layer a neural network layer.
         **/
        void AddLayer(NeuralLayer *layer);
        /**
         * Make a prediction based on the input and the layers.
         * @param input the input to this network.
         * @return a Matrix as the output.
         **/
        Matrix<double> Predict(const Matrix<double> &input);
    };
} // namespace MachineLearning

#endif // NEURALNETWORK_HPP