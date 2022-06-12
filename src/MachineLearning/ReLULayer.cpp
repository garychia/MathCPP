#include "ReLULayer.hpp"

#include <sstream>

namespace MachineLearning
{
    ReLULayer::ReLULayer()
    {
    }

    Matrix<double> ReLULayer::Forward(const Matrix<double> &input)
    {
        this->input = input;
        return this->output = input.Map([](const double &e)
                                        { return e > 0 ? e : 0; });
    }

    Matrix<double> ReLULayer::Backward(const Matrix<double> &derivative)
    {
        return this->output
            .Map([](const double &e)
                 { return e > 0 ? 1 : 0; })
            .Scale(derivative);
    }

    std::string ReLULayer::ToString() const
    {
        std::stringstream ss;
        ss << "ReLULayer:\n";
        ss << "Input:\n"
           << this->input << std::endl;
        ss << "Ouput:\n"
           << this->output;
        return ss.str();
    }
} // namespace MachineLearning