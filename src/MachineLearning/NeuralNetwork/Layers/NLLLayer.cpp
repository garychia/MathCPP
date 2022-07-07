#include "NLLLayer.hpp"
#include "Matrix.hpp"
#include "Math.hpp"

#define LOG_EPSILON 0.0000000000000000001

namespace MachineLearning
{
    NLLLayer::NLLLayer()
    {
    }

    double NLLLayer::ComputeLoss(const DataStructures::Matrix<double> &prediction, const DataStructures::Matrix<double> &labels)
    {
        return -prediction
                    .Map([](const double &e)
                         { return Math::NaturalLog(e + LOG_EPSILON); })
                    .Scale(labels)
                    .SumAll();
    }

    Matrix<double> NLLLayer::Backward(const DataStructures::Matrix<double> &prediction, const DataStructures::Matrix<double> &labels)
    {
        return prediction - labels;
    }

    std::string NLLLayer::ToString() const
    {
        return "NLLLayer";
    }
} // namespace MachineLearning