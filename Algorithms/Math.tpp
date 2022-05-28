#define EPSILON 0.00000001
#define LN_2 0.69314718
#define PI 3.14159265
#define PI_TIMES_2 6.28318531

namespace Math
{
    template <class T>
    T Exponent(const T& x)
    {
        T result = 1;
        T numerator = 1;
        T denominator = 1;
        std::size_t i = 1;
        while (numerator / denominator > EPSILON)
        {
            numerator *= x;
            denominator *= i;
            result += numerator / denominator;
            i++;
        }
        return result;
    }

    template <class T>
    T NaturalLog(const T& x)
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
    T Sine(const T& x)
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
    T Cosine(const T& x)
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
    T Tangent(const T& x)
    {
        return Sine(x) / Cosine(x);
    }

    template <class T, class PowerType>
    T Power(const T& scaler, PowerType n)
    {
        return pow(scaler, n);
    }

    template <class T, class PowerType>
    Vector<T> Power(const Vector<T>& v, PowerType n)
    {
        return v.Map([&n](T e) { return pow(e, n); });
    }

    template <class T, class PowerType>
    Matrix<T> Power(const Matrix<T>& m, PowerType n)
    {
        return m.Map([&n](T e) { return pow(e, n); });
    }

    template <class T>
    T Log(T scaler)
    {
        return std::log(scaler);
    }

    template <class T>
    Vector<T> Log(const Vector<T> &v)
    {
        return v.Map([](T e) { return std::log(e); });
    }

    template <class T>
    Matrix<T> Log(const Matrix<T> &m)
    {
        return m.Map([](T e) { return std::log(e); });
    }

    template <class T>
    T EuclideanNorm(const Vector<T>& v)
    {
        return sqrt(Power<T>(v, 2).Sum());
    }

    template <class T>
    T FrobeniusNorm(const Matrix<T>& m)
    {
        return sqrt(Power<T>(m, 2).Sum());
    }
} // namespace Math
