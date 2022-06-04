#ifndef MATH_HPP
#define MATH_HPP

namespace DataStructure
{
    template <class T>
    class Vector;

    template <class T>
    class Matrix;
}

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
    Calculates the value of exponential e raised to each element of a Vector.
    @param x a Vector.
    @return a Vector with the exponentials.
    */
    template <class T>
    Vector<T> Exponent(const Vector<T>& x);

    /*
    Calculates the value of exponential e raised to each element of a Matrix.
    @param x a Matrix.
    @return a Matrix with the exponentials.
    */
    template <class T>
    Matrix<T> Exponent(const Matrix<T>& x);

    /*
    Computes the natural logarithm given an input.
    @param x a positive value as the input to the natural logarithm.
    @return the natural logarithm.
    */
    template <class T>
    T NaturalLog(const T& x);

    /*
    Computes the natural logarithm of each element of a Vector.
    @param v a Vector.
    @return a Vector with the natural logarithms of its elements.
    */
    template <class T>
    Vector<T> NaturalLog(const Vector<T> &v);

    /*
    Computes the natural logarithm of each element of a Matrix.
    @param m a Matrix.
    @return a Matrix with the natural logarithms of its elements.
    */
    template <class T>
    Matrix<T> NaturalLog(const Matrix<T> &m);

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
    Tangent Function
    @param x the input to the function in radians.
    @return the value of tangent of x.
    */
    template <class T>
    T Tangent(const T& x);

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

    /*
    Rectified Linear Unit Function.
    @param x the input to the function.
    @return the output of ReLU function.
    */
    template <class T>
    T ReLU(const T &x);

    /*
    Sigmoid Function.
    @param x the input to the function.
    @return the output of the function.
    */
    template <class T>
    T Sigmoid(const T &x);

    /*
     * Softmax Function.
     * @param vector a Vector.
     * @return a Vector with its values computed by the function.
     */
    template <class T>
    Vector<T> Softmax(const Vector<T> &vector);
} // namespace Math

#include "Math.tpp"

#endif
