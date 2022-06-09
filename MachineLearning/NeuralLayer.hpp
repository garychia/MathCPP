#ifndef NEURALLAYER_HPP
#define NEURALLAYER_HPP

#include "Matrix.hpp"

#include <ostream>

using namespace DataStructures;

namespace MachineLearning
{
    /**
     * (Dense) Neural Network Layer.
     **/
    class NeuralLayer
    {
    private:
        // Matrix whose columns represent a unit.
        Matrix<double> weights;
        // Column vector whose elements are the biases of the units.
        Matrix<double> biases;
        // Output of this layer.
        Matrix<double> output;

    public:
        // NeuralLayer Contructor.
        NeuralLayer();
        /**
         * NeuralLayer Contructor
         * @param inputSize the dimension of the input.
         * @param outputSize the dimension of the output of this layer.
         **/
        NeuralLayer(std::size_t inputSize, std::size_t outputSize);
        // NeuralLayer Destructor.
        virtual ~NeuralLayer();
        /**
         * Generate an output based on the weights and biases.
         * @param input the input to this layer.
         * @return the output of this layer.
         **/
        Matrix<double> Forward(const Matrix<double> &input);
        /**
         * Generate a string description of this layer.
         * @return a string that describes this layer.
         **/
        std::string ToString() const;

        /**
         * Retrieve the lastest output of this layer.
         * @return the lastest output of this layer. If this layer has
         * not produced any output yet, an empty matrix is returned.
         **/
        Matrix<double> GetLastOutput() const;

        friend std::ostream &operator<<(std::ostream &stream, const NeuralLayer &layer);
    };
} // namespace MachineLearning

#endif // NEURALLAYER_HPP