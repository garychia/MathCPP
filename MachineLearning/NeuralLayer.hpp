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
    protected:
        // Input to this layer.
        Matrix<double> input;
        // Output of this layer.
        Matrix<double> output;

    public:
        // NeuralLayer Contructor.
        NeuralLayer();
        /**
         * Generate an output based on the weights and biases.
         * @param input the input to this layer.
         * @return the output of this layer.
         **/
        virtual Matrix<double> Forward(const Matrix<double> &input) = 0;
        /**
         * Generate a string description of this layer.
         * @return a string that describes this layer.
         **/
        virtual std::string ToString() const = 0;

        /**
         * Retrieve the lastest input to this layer.
         * @return the lastest input to this layer. If this layer has
         * not received any input yet, an empty matrix is returned.
         **/
        Matrix<double> GetLastInput() const;

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