#ifndef MATH_H
#define MATH_H

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

#endif
