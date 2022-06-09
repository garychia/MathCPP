#ifndef LINEARLAYER_HPP
#define LINEARLAYER_HPP

#include "NeuralLayer.hpp"

using namespace DataStructures;

namespace MachineLearning
{
    /**
     * Linear Neural Network Layer.
     **/
    class LinearLayer : public NeuralLayer
    {
    private:
        // Matrix whose columns represent a unit.
        Matrix<double> weights;
        // Column vector whose elements are the biases of the units.
        Matrix<double> biases;

    public:
        // LinearLayer Contructor.
        LinearLayer();
        /**
         * LinearLayer Contructor
         * @param inputSize the dimension of the input.
         * @param outputSize the dimension of the output of this layer.
         **/
        LinearLayer(std::size_t inputSize, std::size_t outputSize);
        // LinearLayer Destructor.
        virtual ~LinearLayer() = default;
        /**
         * Generate an output based on the weights and biases.
         * @param input the input to this layer.
         * @return the output of this layer.
         **/
        Matrix<double> Forward(const Matrix<double> &input) override;
        /**
         * Generate a string description of this layer.
         * @return a string that describes this layer.
         **/
        std::string ToString() const override;
    };
} // namespace MachineLearning

#endif // NEURALLAYER_HPP