#ifndef NEURALLAYER_HPP
#define NEURALLAYER_HPP

#include "Matrix.hpp"

#include <ostream>

using namespace DataStructures;

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
        virtual ~NeuralLayer();
        Matrix<double> Forward(const Matrix<double> &input) const;
        std::string ToString() const;

        friend std::ostream &operator<<(std::ostream &stream, const NeuralLayer &layer)
        {
            stream << layer.ToString();
            return stream;
        }
    };
} // namespace MachineLearning

#endif // NEURALLAYER_HPP