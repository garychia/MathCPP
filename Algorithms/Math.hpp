#ifndef MATH_HPP
#define MATH_HPP

#include "Vector.hpp"
#include "Matrix.hpp"
#include "Exceptions.hpp"

#include <sstream>
#include <cmath>

using namespace DataStructure;

namespace Math
{

    /*
    Calculates the value of exponential e raised to a given number.
    @param x the power.
    @return the exponential.
    */
    template <class T>
    T Exponent(const T& x);

    /*
    Computes the natural logarithm given an input.
    @param x a positive value as the input to the natural logarithm.
    @return the natural logarithm.
    */
    template <class T>
    T NaturalLog(const T& x);

    /*
    Sine Function
    @param x the input to the function in radians.
    @return the value of sine of x.
    */
    template <class T>
    T Sine(const T& x);

    /*
    Cosine Function
    @param x the input to the function in radians.
    @return the value of cosine of x.
    */
    template <class T>
    T Cosine(const T& x);

    /*
    Calculates the power of a scaler.
    @param scaler a scaler.
    @param n the exponent.
    @return the power of the scaler.
    */
    template <class T, class PowerType>
    T Power(const T& scaler, PowerType n);

    /*
    Calculates the power of each element of a Vector.
    @param v a Vector.
    @param n the exponent.
    @return a Vector with the powers of its elements.
    */
    template <class T, class PowerType>
    Vector<T> Power(const Vector<T>& v, PowerType n);

    /*
    Calculates the power of each element of a Matrix.
    @param v a Matrix.
    @param n the exponent.
    @return a Matrix with the powers of its elements.
    */
    template <class T, class PowerType>
    Matrix<T> Power(const Matrix<T>& m, PowerType n);

    /*
    Computes the natural logarithm.
    @param scaler a scaler.
    @return the natural logarithm of the given scaler.
    */
    template <class T>
    T Log(T scaler);

    /*
    Computes the natural logarithm of each element of a Vector.
    @param v a Vector.
    @return a Vector with the natural logarithms of its elements.
    */
    template <class T>
    Vector<T> Log(const Vector<T> &v);

    /*
    Computes the natural logarithm of each element of a Matrix.
    @param m a Matrix.
    @return a Matrix with the natural logarithms of its elements.
    */
    template <class T>
    Matrix<T> Log(const Matrix<T> &m);

    /*
    Calculates the Euclidean norm of a Vector.
    @param v a Vector.
    @return the Euclidean norm of the given Vector.
    */
    template <class T>
    T EuclideanNorm(const Vector<T>& v);

    /*
    Calculates the Frobenius norm of a Matrix.
    @param m a Matrix.
    @return the Frobenius norm of the given Matrix.
    */
    template <class T>
    T FrobeniusNorm(const Matrix<T>& m);
} // namespace Math

#include "Math.tpp"

#endif
