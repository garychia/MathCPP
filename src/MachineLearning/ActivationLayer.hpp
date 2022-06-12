#ifndef ACTIVATIONLAYER_HPP
#define ACTIVATIONLAYER_HPP

#include "NeuralLayer.hpp"

namespace MachineLearning
{
    class ActivationLayer : public NeuralLayer
    {
    public:
        ActivationLayer();
        virtual ~ActivationLayer() = default;
        virtual void UpdateWeights(const double &learningRate) override;
    };
} // namespace MachineLearning

#endif