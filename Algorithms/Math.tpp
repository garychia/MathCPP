#include "Vector.hpp"
#include "Matrix.hpp"
#include "Exceptions.hpp"

#include <sstream>

#define EPSILON 0.00000001
#define LN_2 0.69314718
#define PI 3.14159265
#define PI_TIMES_2 6.28318531

namespace Math
{
    template <class T>
    T Exponent(const T &x)
    {
        const T input = x < 0 ? -x : x;
        T result = 1;
        T numerator = input;
        T denominator = 1;
        std::size_t i = 2;
        while (numerator / denominator > EPSILON)
        {
            result += numerator / denominator;
            numerator *= input;
            denominator *= i;
            i++;
        }
        return x < 0 ? 1 / result : result;
    }

    template <class T>
    Vector<T> Exponent(const Vector<T> &x)
    {
        return x.Map([](T e)
                     { return Exponent(e); });
    }

    template <class T>
    Matrix<T> Exponent(const Matrix<T> &x)
    {
        return x.Map([](T e)
                     { return Exponent(e); });
    }

    template <class T>
    T NaturalLog(const T &x)
    {
        if (x <= 0)
        {
            std::stringstream ss;
            ss << "NaturalLog: Input must be positive but got "
               << x
               << ".";
            throw Exceptions::InvalidArgument(ss.str());
        }
        std::size_t exponent = 0;
        T input = x;
        while (input >= 2)
        {
            input /= 2;
            exponent++;
        }
        T result = 0;
        T factor = 1;
        T numerator = input - 1;
        std::size_t denominator = 1;
        while (numerator / denominator > EPSILON)
        {
            result += factor * numerator / denominator;
            factor = -factor;
            numerator *= input - 1;
            denominator++;
        }
        return result + exponent * LN_2;
    }

    template <class T>
    Vector<T> NaturalLog(const Vector<T> &v)
    {
        return v.Map([](T e)
                     { return NaturalLog(e); });
    }

    template <class T>
    Matrix<T> NaturalLog(const Matrix<T> &m)
    {
        return m.Map([](T e)
                     { return NaturalLog(e); });
    }

    template <class T>
    T Sine(const T &x)
    {
        T input = x < 0 ? -x : x;
        while (input >= PI_TIMES_2)
            input -= PI_TIMES_2;
        T squaredInput = input * input;
        T factor = 1;
        T numerator = input;
        T denominator = 1;
        T result = numerator / denominator;
        std::size_t i = 3;
        while (numerator / denominator > EPSILON)
        {
            factor = -factor;
            numerator *= squaredInput;
            denominator *= i * (i - 1);
            i += 2;
            result += factor * numerator / denominator;
        }
        return x < 0 ? -result : result;
    }

    template <class T>
    T Cosine(const T &x)
    {
        T input = x < 0 ? -x : x;
        while (input >= PI_TIMES_2)
            input -= PI_TIMES_2;
        T squaredInput = input * input;
        T factor = 1;
        T numerator = 1;
        T denominator = 1;
        T result = numerator / denominator;
        std::size_t i = 2;
        while (numerator / denominator > EPSILON)
        {
            factor = -factor;
            numerator *= squaredInput;
            denominator *= i * (i - 1);
            i += 2;
            result += factor * numerator / denominator;
        }
        return result;
    }

    template <class T>
    T Tangent(const T &x)
    {
        return Sine(x) / Cosine(x);
    }

    template <class T>
    T Sinh(const T &x)
    {
        const T exponential = Exponent(x);
        return (exponential - 1 / exponential) * 0.5;
    }

    template <class T>
    T Cosh(const T &x)
    {
        const T exponential = Exponent(x);
        return (exponential + 1 / exponential) * 0.5;
    }

    template <class T>
    T Tanh(const T &x)
    {
        if (2 * x > 14 || 2 * x < -14)
            return x > 0 ? 1 : -1;
        const T exponential = Exponent(2 * x);
        return (exponential - 1) / (exponential + 1);
    }

    template <class T, class PowerType>
    T Power(const T &scaler, PowerType n)
    {
        if (scaler > 0)
            return Exponent(n * NaturalLog(scaler));
        else if (scaler == 0)
            return 0;
        else if ((long)n % 2 == 0)
            return Exponent(n * NaturalLog(-scaler));
        return -Exponent(n * NaturalLog(scaler));
    }

    template <class T, class PowerType>
    Vector<T> Power(const Vector<T> &v, PowerType n)
    {
        return v.Map([&n](T e)
                     { return Power(e, n); });
    }

    template <class T, class PowerType>
    Matrix<T> Power(const Matrix<T> &m, PowerType n)
    {
        return m.Map([&n](T e)
                     { return Power(e, n); });
    }

    template <class T>
    T EuclideanNorm(const Vector<T> &v)
    {
        return sqrt(Power<T>(v, 2).Sum());
    }

    template <class T>
    T FrobeniusNorm(const Matrix<T> &m)
    {
        return sqrt(Power<T>(m, 2).Sum());
    }

    template <class T>
    T ReLU(const T &x)
    {
        if (x < 0)
            return 0;
        return x;
    }

    template <class T>
    T Sigmoid(const T &x)
    {
        return 1 / (1 + Exponent(-x));
    }

    template <class T>
    Vector<T> Softmax(const Vector<T> &vector)
    {
        const T denomerator = Exponent(vector).Sum();
        const Vector<T> numerator = Exponent(vector);
        return numerator / denomerator;
    }

    template <class T>
    Matrix<T> Softmax(const Matrix<T> &matrix)
    {
        const auto exponents = Exponent(matrix);
        const auto summation = exponents.Sum();
        return exponents / summation;
    }

    template <class T>
    T Gauss(const T &x, const T &mu, const T &sigma)
    {
        const T normalization = (x - mu) / sigma;
        return 1 / (sigma * Power(2 * PI, 0.5)) * Exponent(-0.5 * normalization * normalization);
    }
} // namespace Math
