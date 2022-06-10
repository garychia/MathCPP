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
        enum class LayerType
        {
            Linear,
            ReLU,
            SoftMax,
            Tanh
        };
        
        /* NeuralNetwork Constructor */
        NeuralNetwork();
        /**
         * NeuralNetwork Constructor
         * @param layerTypes a list of LayerType to specify the type of each layer.
         * @param shapes a list of Tuple to specify the input and output size of each layer.
         **/
        NeuralNetwork(const List<LayerType> &layerTypes, const List<Tuple<unsigned int>> &shapes);
        // NeuralNetwork Destructor
        ~NeuralNetwork();
        /**
         * Make a prediction based on the input and the layers.
         * @param input the input to this network.
         * @return a Matrix as the output.
         **/
        Matrix<double> Predict(const Matrix<double> &input);
    };

    std::ostream &operator<<(std::ostream &stream, const NeuralNetwork::LayerType &layerType);
} // namespace MachineLearning

#endif // NEURALNETWORK_HPP