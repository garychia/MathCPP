#ifndef MATH_HPP
#define MATH_HPP

#include "../DataStructures/vector.hpp"
#include "../DataStructures/matrix.hpp"

#include <cmath>

using namespace DataStructure;

namespace Math
{
    /*
    Calculates the power of a scaler.
    @param scaler a scaler.
    @param n the exponent.
    @return the power of the scaler.
    */
    template <class T>
    T Power(const T& scaler, T n)
    {
        return pow(scaler, n);
    }

    /*
    Calculates the power of each element of a Vector.
    @param v a Vector.
    @param n the exponent.
    @return a Vector with the powers of its elements.
    */
    template <class T>
    Vector<T> Power(const Vector<T>& v, T n)
    {
        return v.Map([&n](T e) { return pow(e, n); });
    }

    /*
    Calculates the power of each element of a Matrix.
    @param v a Matrix.
    @param n the exponent.
    @return a Matrix with the powers of its elements.
    */
    template <class T>
    Matrix<T> Power(const Matrix<T>& m, T n)
    {
        return m.Map([&n](T e) { return pow(e, n); });
    }

    /*
    Computes the natural logarithm.
    @param scaler a scaler.
    @return the natural logarithm of the given scaler.
    */
    template <class T>
    T Log(T scaler)
    {
        return std::log(scaler);
    }

    /*
    Computes the natural logarithm of each element of a Vector.
    @param v a Vector.
    @return a Vector with the natural logarithms of its elements.
    */
    template <class T>
    Vector<T> Log(const Vector<T> &v)
    {
        return v.Map([](T e) { return std::log(e); });
    }

    /*
    Computes the natural logarithm of each element of a Matrix.
    @param m a Matrix.
    @return a Matrix with the natural logarithms of its elements.
    */
    template <class T>
    Matrix<T> Log(const Matrix<T> &m)
    {
        return m.Map([](T e) { return std::log(e); });
    }

    /*
    Calculates the Euclidean norm of a Vector.
    @param v a Vector.
    @return the Euclidean norm of the given Vector.
    */
    template <class T>
    T EuclideanNorm(const Vector<T>& v)
    {
        return sqrt(Power<T>(v, 2).Sum());
    }

    /*
    Calculates the Frobenius norm of a Matrix.
    @param m a Matrix.
    @return the Frobenius norm of the given Matrix.
    */
    template <class T>
    T FrobeniusNorm(const Matrix<T>& m)
    {
        return sqrt(Power<T>(m, 2).Sum());
    }
} // namespace Math

#endif
