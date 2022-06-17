#include "Vector.hpp"
#include "Matrix.hpp"
#include "List.hpp"
#include "Exceptions.hpp"

#include <sstream>

#define EPSILON 0.00000000001
#define LN_2 0.69314718056
#define PI 3.14159265359
#define PI_TIMES_2 6.28318530718
#define ABS(x) ((x) >= 0 ? (x) : -(x))

namespace Math
{
    template <class T>
    T Exponent(const T &x)
    {
        if (x == 0)
            return 1;
        const auto input = ABS(x);
        T result = 0;
        T numerator = 1;
        std::size_t denominator = 1;
        std::size_t i = 1;
        T term = numerator / denominator;
        while (ABS(term) > EPSILON)
        {
            result += term;
            if (ABS(numerator) >= 1E10 / input)
            {
                numerator /= denominator;
                denominator = 1;
            }
            numerator *= input;
            denominator *= i;
            i++;
            term = numerator / denominator;
        }
        return x > 0 ? result : 1 / result;
    }

    template <class T>
    T NaturalLog(const T &x)
    {
        if (x <= 0)
            throw Exceptions::InvalidArgument(
                "NaturalLog: Expected the input to be positive.");
        T input = x;
        T exp = 0;
        while (input > 1)
        {
            input /= 2;
            exp++;
        }
        input = input - 1;
        bool positiveTerm = true;
        T result = 0;
        T numerator = input;
        T denominator = 1;
        T ratio = numerator / denominator;
        for (std::size_t i = 0; i < 1000; i++)
        {
            result += ratio * (positiveTerm ? 1 : -1);
            numerator *= input;
            denominator++;
            ratio = numerator / denominator;
            positiveTerm = !positiveTerm;
        }
        return result + LN_2 * exp;
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

    template <class T>
    T _PowerLong(const T &scaler, long n)
    {
        if (scaler == 0 && n <= 0)
        {
            std::stringstream ss;
            ss << "Power: " << scaler;
            ss << " to the power of " << n;
            ss << " is undefined.";
            throw Exceptions::InvalidArgument(ss.str());
        }
        else if (scaler == 0 || scaler == 1)
            return scaler;
        else if (n == 0)
            return 1;
        auto p = n > 0 ? n : -n;
        DataStructures::List<bool> even;
        while (p > 1)
        {
            even.Append((p & 1) == 0);
            p >>= 1;
        }
        auto result = scaler;
        if (!even.IsEmpty())
        {
            std::size_t i = even.Size();
            while (i != 0)
            {
                result *= result;
                if (!even[i - 1])
                    result *= scaler;
                i--;
            }
        }
        return n > 0 ? result : T(1) / result;
    }

    template <class T, class PowerType>
    T Power(const T &scaler, PowerType n)
    {
        if (scaler == 0 && n > 0)
            return 0;
        else if (scaler > 0)
        {
            if (n == 0)
                return 1;
            else if (n == 1)
                return scaler;
            else if (n < 0)
                return T(1) / Power(scaler, -n);
            return Exponent(n * NaturalLog(scaler));
        }
        else if (scaler < 0 && (long)n == n)
            return _PowerLong<T>(scaler, (long)n);
        std::stringstream ss;
        ss << "Power: " << scaler;
        ss << " to the power of " << n;
        ss << " is undefined.";
        throw Exceptions::InvalidArgument(ss.str());
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
