#ifndef NLLLayer_HPP
#define NLLLayer_HPP

#include "LossLayer.hpp"

namespace MachineLearning
{
    class NLLLayer : public LossLayer
    {
    public:
        NLLLayer();
        ~NLLLayer() = default;
        /**
         * Compute the loss based on the prediction and labels.
         * @param prediction the prediction a neural network has generated.
         * @param labels the correct output (0's and 1's) the network should generate.
         * @return the loss.
         **/
        virtual double ComputeLoss(const DataStructures::Matrix<double> &prediction, const DataStructures::Matrix<double> &labels) override;
        /**
         * Compute the derivative with respect to the prediction a neural network has generated.
         * @param prediction the prediction from the network.
         * @param labels the correct output (0's and 1's) the network should generate.
         * @return the derivative.
         **/
        virtual DataStructures::Matrix<double> Backward(const DataStructures::Matrix<double> &prediction, const DataStructures::Matrix<double> &labels) override;

        /**
         * Generate a string that describes this NLLLayer.
         * @return a string that describes this NLLLayer.
         **/
        virtual std::string ToString() const override;
    };
} // namespace MachineLearning

#endif // NLLLayer_HPP