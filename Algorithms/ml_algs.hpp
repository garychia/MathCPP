#include "../DataStructures/vector.hpp"
#include "../DataStructures/matrix.hpp"

using namespace DataStructure;

namespace MLAlgs
{
    /*
    Sign function.
    @param value a value.
    @return +1 if the value is positive, or 0 if the value is 0. -1, otherwise.
    */
    template <class T>
    T Sign(T value)
    {
        return value == 0 ? 0 : (value > 0 ? 1 : -1);
    }

    /*
    The Perceptron Algorithm.
    @param data a Matrix with n rows and d columns where n is the number of data points
    and d is the dimensions of them.
    @param labels a Vector with a single row and n columns indicating the labels of the
    data points.
    @param T the number of iterations to repeat the algorithm (default: 1000).
    @return a Vector packed with the parameters (theta and offset at the end).
    */
    template <class DataType, class LabelType>
    Vector<double> Perceptron(const Matrix<DataType>& data, const Vector<LabelType>& labels, std::size_t T = 1000)
    {
        const auto dataShape = data.Shape();
        const auto n = dataShape[0];
        const auto d = dataShape[1];
        Vector<double> th(d, 0.0);
        double th0 = 0.0;
        for (std::size_t i = 0; i < T; i++)
            for (std::size_t j = 0; j < n; j++)
                if (Sign(th.Dot(data[j]) + th0) != labels[j])
                {
                    th += data[j] * labels[j];
                    th0 += labels[j];
                }
        return Vector<double>::Combine({std::move(th), Vector<double>(1, th0)});
    }
} // namespace MLAlgs
