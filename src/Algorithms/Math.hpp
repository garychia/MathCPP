#ifndef MATH_HPP
#define MATH_HPP

#include <sstream>

namespace DataStructures {
template <class T> class List;
template <class T> class Matrix;
template <class T> class Vector;
} // namespace DataStructures

namespace Math {
class Constants {
public:
  static const double Epsilon;
  static const double Ln2;
  static const double Pi;
};

template <class T> T Abs(const T &x) { return x >= 0 ? x : -x; }

template <class T> T &Max(const T &x, const T &y) { return x < y ? y : x; }

template <class T> T &Min(const T &x, const T &y) { return x > y ? y : x; }

/*
Calculates the value of exponential e raised to a given number.
@param x the power.
@return the exponential.
*/
template <class T> T Exponent(const T &x);

/*
Computes the natural logarithm given an input.
@param x a positive value as the input to the natural logarithm.
@return the natural logarithm.
*/
template <class T> T NaturalLog(const T &x);

/*
Sine Function
@param x the input to the function in radians.
@return the value of sine of x.
*/
template <class T> T Sine(const T &x);

/*
Cosine Function
@param x the input to the function in radians.
@return the value of cosine of x.
*/
template <class T> T Cosine(const T &x);

/*
Tangent Function
@param x the input to the function in radians.
@return the value of tangent of x.
*/
template <class T> T Tangent(const T &x);

/*
Hyperbolic Sine Function
@param x the input to the function.
@return the value of sinh of x.
*/
template <class T> T Sinh(const T &x);

/*
Hyperbolic Cosine Function
@param x the input to the function.
@return the value of cosh of x.
*/
template <class T> T Cosh(const T &x);

/*
Hyperbolic Tangent Function
@param x the input to the function.
@return the value of tanh of x.
*/
template <class T> T Tanh(const T &x);

/*
Calculates the power of a scaler.
@param scaler a scaler.
@param n the exponent.
@return the power of the scaler.
*/
template <class T, class PowerType> T Power(const T scaler, PowerType n);

/*
Rectified Linear Unit Function.
@param x the input to the function.
@return the output of ReLU function.
*/
template <class T> T ReLU(const T &x);

/*
Sigmoid Function.
@param x the input to the function.
@return the output of the function.
*/
template <class T> T Sigmoid(const T &x);

/*
 * Softmax Function.
 * @param vector a Vector.
 * @return a Vector with its values computed by the function.
 */
template <class T>
DataStructures::Vector<T> Softmax(const DataStructures::Vector<T> &vector);

/*
 * Softmax Function.
 * @param matrix a Matrix.
 * @return a Matrix with its column vectors computed by the function.
 */
template <class T>
DataStructures::Matrix<T> Softmax(const DataStructures::Matrix<T> &matrix);

/*
 * Gaussian Probability Density Function
 * @param x the input to the function.
 * @param mu the mean (average).
 * @param sigma the standard deviation.
 */
template <class T> T Gauss(const T &x, const T &mu, const T &sigma);

} // namespace Math

#include "Exceptions.hpp"
#include "List.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"

namespace Math {
template <class T> T Exponent(const T &x) {
  if (x == 0)
    return 1;
  const auto input = Abs(x);
  T result = 0;
  T numerator = 1;
  std::size_t denominator = 1;
  std::size_t i = 1;
  T term = numerator / denominator;
  while (i < 501 && term >= 1E-20) {
    result += term;
    if (denominator >= 1000) {
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

template <class T> T NaturalLog(const T &x) {
  if (x <= 0)
    throw Exceptions::InvalidArgument(
        "NaturalLog: Expected the input to be positive.");
  T input = x;
  T exp = 0;
  while (input > 1 && exp < 10000) {
    input /= 2;
    exp++;
  }
  input = input - 1;
  bool positiveTerm = true;
  T result = 0;
  T numerator = input;
  T denominator = 1;
  T ratio = numerator / denominator;
  for (std::size_t i = 0; i < 1000; i++) {
    result += ratio * (positiveTerm ? 1 : -1);
    numerator *= input;
    denominator++;
    ratio = numerator / denominator;
    positiveTerm = !positiveTerm;
  }
  return result + (Constants::Ln2)*exp;
}

template <class T> T Sine(const T &x) {
  T input = x < 0 ? -x : x;
  const auto doublePI = Constants::Pi * 2;
  while (input >= doublePI)
    input -= doublePI;
  while (input <= -doublePI)
    input += doublePI;
  T squaredInput = input * input;
  T factor = 1;
  T numerator = input;
  T denominator = 1;
  T result = numerator / denominator;
  std::size_t i = 3;
  while (i < 2006) {
    factor = -factor;
    numerator *= squaredInput;
    denominator *= i * (i - 1);
    i += 2;
    result += factor * numerator / denominator;
  }
  return x < 0 ? -result : result;
}

template <class T> T Cosine(const T &x) {
  T input = x < 0 ? -x : x;
  const auto doublePI = Constants::Pi * 2;
  while (input >= doublePI)
    input -= doublePI;
  while (input <= -doublePI)
    input += doublePI;
  T squaredInput = input * input;
  T factor = 1;
  T numerator = 1;
  T denominator = 1;
  T result = numerator / denominator;
  std::size_t i = 2;
  while (i < 2003) {
    factor = -factor;
    numerator *= squaredInput;
    denominator *= i * (i - 1);
    i += 2;
    result += factor * numerator / denominator;
  }
  return result;
}

template <class T> T Tangent(const T &x) { return Sine(x) / Cosine(x); }

template <class T> T Sinh(const T &x) {
  const T exponential = Exponent(x);
  return (exponential - 1 / exponential) * 0.5;
}

template <class T> T Cosh(const T &x) {
  const T exponential = Exponent(x);
  return (exponential + 1 / exponential) * 0.5;
}

template <class T> T Tanh(const T &x) {
  if (2 * x > 14 || 2 * x < -14)
    return x > 0 ? 1 : -1;
  const T exponential = Exponent(2 * x);
  return (exponential - 1) / (exponential + 1);
}

namespace {
template <class T> T _PowerLong(const T &scaler, long n) {
  if (scaler == 0 && n <= 0) {
    std::stringstream ss;
    ss << "Power: " << scaler;
    ss << " to the power of " << n;
    ss << " is undefined.";
    throw Exceptions::InvalidArgument(ss.str());
  } else if (scaler == 0 || scaler == 1)
    return scaler;
  else if (n == 0)
    return 1;
  auto p = n > 0 ? n : -n;
  DataStructures::List<bool> even;
  while (p > 1) {
    even.Append((p & 1) == 0);
    p >>= 1;
  }
  auto result = scaler;
  if (!even.IsEmpty()) {
    std::size_t i = even.Size();
    while (i != 0) {
      result *= result;
      if (!even[i - 1])
        result *= scaler;
      i--;
    }
  }
  return n > 0 ? result : T(1) / result;
}
} // namespace

template <class T, class PowerType> T Power(const T scaler, PowerType n) {
  if (scaler == 0 && n > 0)
    return 0;
  else if (scaler > 0) {
    if (n == 0)
      return 1;
    else if (n == 1)
      return scaler;
    else if (n < 0)
      return T(1) / Power(scaler, -n);
    return Exponent(n * NaturalLog(scaler));
  } else if (scaler < 0 && (long)n == n)
    return _PowerLong<T>(scaler, (long)n);
  std::stringstream ss;
  ss << "Power: " << scaler;
  ss << " to the power of " << n;
  ss << " is undefined.";
  throw Exceptions::InvalidArgument(ss.str());
}

template <class T> T ReLU(const T &x) { return x < 0 ? 0 : x; }

template <class T> T Sigmoid(const T &x) { return 1 / (1 + Exponent(-x)); }

template <class T>
DataStructures::Vector<T> Softmax(const DataStructures::Vector<T> &vector) {
  const T denomerator = Exponent(vector).Sum();
  const DataStructures::Vector<T> numerator = Exponent(vector);
  return numerator / denomerator;
}

template <class T>
DataStructures::Matrix<T> Softmax(const DataStructures::Matrix<T> &matrix) {
  const auto exponents = Exponent(matrix);
  const auto summation = exponents.Sum();
  return exponents / summation;
}

template <class T> T Gauss(const T &x, const T &mu, const T &sigma) {
  const T normalization = (x - mu) / sigma;
  return 1 / (sigma * Power(2 * Constants::Pi, 0.5)) *
         Exponent(-0.5 * normalization * normalization);
}
} // namespace Math

#endif
