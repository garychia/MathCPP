#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <array>
#include <initializer_list>
#include <vector>

#include "Tuple.hpp"
#include "Math.hpp"

namespace DataStructures {
/* A mutable Container of a fixed size that supports numerical operations. */
template <class T> class Vector : public Tuple<T> {
public:
  /* Constructor that Constructs an Empty Vector. */
  Vector() : Tuple<T>() {}

  /**
   * Constructor with Initial Size and a Value.
   * @param s the initial size of the Vector to be generated.
   * @param value the value the Vector will be filled with.
   **/
  Vector(size_t s, const T &value = 0) : Tuple<T>(s, value) {}

  /**
   * Constructor with a std::initializer_list as Input.
   * @param l a std::initializer_list that contains the elements the Vector will
   *store.
   **/
  Vector(const std::initializer_list<T> &l) : Tuple<T>(l) {}

  /**
   * Constructor with a std::arrary as Input.
   * @param arr a std::array that contains the elements the Vector will store.
   **/
  template <size_t N> Vector(const std::array<T, N> &arr) : Tuple<T>(arr) {}

  /**
   * Constructor with a std::vector as Input.
   * @param v a std::vector that contains the elements the Vector will store.
   **/
  Vector(const std::vector<T> &v) : Tuple<T>(v) {}

  /**
   * Copy Constructor.
   * @param other a Container whose elements will be copied into the Vector.
   **/
  Vector(const Tuple<T> &other) : Tuple<T>(other) {}

  /**
   * Copy Constructor.
   * @param other a Container to be copied.
   **/
  template <class OtherType>
  Vector(const Tuple<OtherType> &other) : Tuple<T>(other) {}

  /**
   * Move Constructor.
   * @param other a Container whose elements will be 'moved' into the Vector.
   **/
  Vector(Tuple<T> &&other) : Tuple<T>(std::move(other)) {}

  /**
   * Copy Assignment.
   * @param other a Container whose elements will be copied into this Vector.
   * @return a reference to this Vector.
   **/
  virtual Vector<T> &operator=(const Tuple<T> &other) {
    Tuple<T>::operator=(other);
    return *this;
  }

  /**
   * Copy Assignment.
   * @param other a Container that contains values of a different type.
   * @return a reference to this Vector.
   **/
  template <class OtherType>
  Vector<T> &operator=(const Tuple<OtherType> &other) {
    Tuple<T>::operator=(other);
    return *this;
  }

  Vector<T> &operator=(Tuple<T> &&other) {
    Tuple<T>::operator=(other);
    return *this;
  }

  /**
   * Access the element at a given index.
   * @param index the index at which the element will be accessed.
   * @return a reference to the accessed element.
   * @throw IndexOutOfBound when the index exceeds the largest possible index.
   **/
  virtual T &operator[](const size_t &index) {
    if (index < Size())
      return this->data[index];
    throw Exceptions::IndexOutOfBound(
        index, "Vector: Index must be less than the dimension.");
  }

  /**
   * Access the element at a given index.
   * @param index the index at which the element will be accessed.
   * @return a reference to the accessed element.
   * @throw IndexOutOfBound when the index exceeds the largest possible index.
   **/
  virtual const T &operator[](const size_t &index) const override {
    if (index < Size())
      return this->data[index];
    throw Exceptions::IndexOutOfBound(
        index, "Vector: Index must be less than the dimension.");
  }

  virtual bool IsEmpty() const { return Dimension() == 0; }

  /**
   * Return the dimention (number of values) of the Vector.
   * @return the dimention.
   **/
  size_t Dimension() const { return Size(); }

  /**
   * Return the Euclidean norm of the Vector.
   * @return the Euclidean norm.
   * @throw EmptyVector if the Vector is empty.
   **/
  template <class ReturnType> ReturnType Length() const {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Length of an empty vector is undefined.");
    return LpNorm<ReturnType>(2);
  }

  /**
   * Return the Euclidean norm of the Vector. (Same as Vector::Length)
   * @return the Euclidean norm.
   * @throw EmptyVector if the Vector is empty.
   **/
  template <class ReturnType> ReturnType EuclideanNorm() const {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Euclidean norm of an empty vector is undefined.");
    return Length<ReturnType>();
  }

  /**
   * Returns the Lp Norm of the Vector.
   * @return the Lp norm.
   * @throw EmptyVector if the Vector is empty.
   **/
  template <class ReturnType> ReturnType LpNorm(int p) const {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Lp norm of an empty vector is undefined.");
    ReturnType squaredTotal = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : squaredTotal)
    for (size_t i = 0; i < Dimension(); i++)
      squaredTotal += Math::template Power<ReturnType, int>((*this)[i], p);
    return Math::template Power<ReturnType, double>(squaredTotal, (double)1 / p);
  }

  /**
   * Perform addition with two Vectors.
   * @param other a Vector as the second operand.
   * @return a new Vector that is the sum of the Vectors.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty or the number of
   *elements of the second operand is not a factor of that of this Vector.
   **/
  template <class OtherType> auto Add(const Vector<OtherType> &other) const {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform addition on an empty vector.");
    else if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Vector: Cannot perform addtion on the given empty vector.");
    else if (Dimension() % other.Dimension() != 0)
      throw Exceptions::InvalidArgument(
          "Vector: Expected the dimension of the second operand to be a factor "
          "of that of the first operand.");
    Vector<decltype(this->data[0] + other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      result[i] += other[i % other.Dimension()];
    return result;
  }

  /**
   * Perform element-wise addition with a Vector and a scaler.
   * @param scaler a scaler to be added to each element of the Vector.
   * @return a new Vector as the addition result.
   * @throw EmptyVector if the Vector is empty.
   **/
  template <class ScalerType> auto Add(const ScalerType &scaler) const {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform addition on an empty vector.");
    Vector<decltype(this->data[0] + scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      result[i] += scaler;
    return result;
  }

  /**
   * Perform addition with two Vectors. (See Vector::Add)
   * @param other a Vector to be added to this Vector.
   * @return a new Vector as the addition result.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty or the number of
   *elements of the second operand is not a factor of that of this Vector.
   **/
  template <class OtherType>
  auto operator+(const Vector<OtherType> &other) const {
    return this->Add(other);
  }

  /**
   * Perform element-wise addition with a Vector and a scaler. (See Vector::Add)
   * @param scaler a scaler to be added to each element of the Vector.
   * @return a new Vector as the addition result.
   * @throw EmptyVector if the Vector is empty.
   **/
  template <class ScalerType> auto operator+(const ScalerType &scaler) const {
    return this->Add(scaler);
  }

  /**
   * Perform inplace addition with another Vector.
   * @param other a Vector to be added to this Vector.
   * @return the reference to this Vector.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty or the number of
   *elements of the second operand is not a factor of that of this Vector.
   **/
  template <class OtherType>
  Vector<T> &operator+=(const Vector<OtherType> &other) {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform addition on an empty vector.");
    else if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Vector: Cannot perform addtion on the given empty vector.");
    else if (Dimension() % other.Dimension() != 0)
      throw Exceptions::InvalidArgument(
          "Vector: Expected the dimension of the second operand to be a factor "
          "of that of the first operand.");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      this->data[i] += other[i % other.Dimension()];
    return *this;
  }

  /**
   * Perform inplace addition with a scaler.
   * @param scaler a scaler to be added to each element of the Vector.
   * @return the reference to this Vector.
   * @throw EmptyVector if this Vector is empty.
   **/
  template <class ScalerType> Vector<T> &operator+=(const ScalerType &scaler) {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform addition on an empty vector.");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      this->data[i] += scaler;
    return *this;
  }

  /**
   * Perform subtraction with two Vectors.
   * @param other a Vector as the second operand.
   * @return a new Vector as the subtraction result.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty or the number of
   *elements of the second operand is not a factor of that of this Vector.
   **/
  template <class OtherType> auto Minus(const Vector<OtherType> &other) const {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform subtraction on an empty vector.");
    else if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Vector: Cannot perform subtraction on the given empty vector.");
    else if (Dimension() % other.Dimension() != 0)
      throw Exceptions::InvalidArgument(
          "Vector: Expected the dimension of the second operand to be a factor "
          "of that of the first operand.");
    Vector<decltype(this->data[0] - other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      result[i] -= other[i % other.Dimension()];
    return result;
  }

  /**
   * Perform element-wise subtraction with a scaler.
   * @param scaler a scaler to be subtracted from each element of the Vector.
   * @return a Vector as the subtraction result.
   * @throw EmptyVector if the Vector is empty.
   **/
  template <class ScalerType> auto Minus(const ScalerType &scaler) const {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform subtraction on an empty vector.");
    Vector<decltype(this->data[0] - scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      result[i] -= scaler;
    return result;
  }

  /**
   * Perform subtraction with two Vectors. (See Vector::Minus)
   * @param other a Vector as the second operand.
   * @return a new Vector as the subtraction result.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty or the number of
   *elements of the second operand is not a factor of that of this Vector.
   **/
  template <class OtherType>
  auto operator-(const Vector<OtherType> &other) const {
    return this->Minus(other);
  }

  /**
   * Perform subtraction with a Vector and a scaler. (See Vector.Minus)
   * @param scaler a scaler to be subtracted from each element of the Vector.
   * @return a new Vector that is the subtraction result.
   * @throw EmptyVector if this Vector is empty.
   **/
  template <class ScalerType> auto operator-(const ScalerType &scaler) const {
    return this->Minus(scaler);
  }

  /**
   * Perform inplace subtraction with another Vector.
   * @param other a Vector to be subtracted.
   * @return the reference to this Vector.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty or the number of
   *elements of the second operand is not a factor of that of this Vector.
   **/
  template <class OtherType>
  Vector<T> &operator-=(const Vector<OtherType> &other) {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform subtraction on an empty vector.");
    else if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Vector: Cannot perform subtraction on the given empty vector.");
    else if (Dimension() % other.Dimension() != 0)
      throw Exceptions::InvalidArgument(
          "Vector: Expected the dimension of the second operand to be a factor "
          "of that of the first operand.");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      this->data[i] -= other[i % other.Dimension()];
    return *this;
  }

  /**
   * Perform inplace subtraction with a scaler.
   * @param scaler a scaler to be subtracted from each element of the Vector.
   * @return the reference to the Vector.
   * @throw EmptyVector if the Vector is empty.
   **/
  template <class ScalerType> Vector<T> &operator-=(const ScalerType &scaler) {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform subtraction on an empty vector.");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      this->data[i] -= scaler;
    return *this;
  }

  /**
   * Perform element-wise multiplication with a Vector and a scaler.
   * @param scaler a scaler to be multiplied with each element of the Vector.
   * @return a new Vector as the multiplication result.
   * @throw EmptyVector if this Vector is empty.
   **/
  template <class ScalerType> auto Scale(const ScalerType &scaler) const {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform scaling on an empty vector.");
    Vector<decltype(this->data[0] * scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      result[i] *= scaler;
    return result;
  }

  /**
   * Perform element-wise multiplication with two Vectors.
   * @param other a Vector as the second operand.
   * @return a new Vector as the element-wise multiplication result.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty.
   * @throw InvalidArgument if the dimension of the second operand is not a
   * factor of the first operand.
   **/
  template <class OtherType> auto Scale(const Vector<OtherType> &other) const {
    if (Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform scaling on an empty vector.");
    else if (other.Dimension() == 0)
      throw Exceptions::InvalidArgument(
          "Vector: Cannot perform scaling with an "
          "empty vector as the second operand.");
    else if (Dimension() % other.Dimension() != 0)
      throw Exceptions::InvalidArgument(
          "Vector: Expected the dimension of the second operand to be a factor "
          "of that of the first operand.");
    Vector<decltype((*this)[0] * other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      result[i] *= other[i % other.Dimension()];
    return result;
  }

  /**
   * Perform element-wise multiplication with a Vector and a scaler. (See
   *Vector::Scale)
   * @param scaler a scaler used to scale this Vector.
   * @return a new Vector as the multiplication result.
   * @throw EmptyVector if the Vector is empty.
   **/
  template <class ScalerType> auto operator*(const ScalerType &scaler) const {
    return this->Scale(scaler);
  }

  /**
   * Perform element-wise multiplication with two Vectors.
   * @param other a Vector.
   * @return a Vector that is the result of the multiplication.
   * @throw EmptyVector when this vector is empty.
   * @throw InvalidArgument if the second operand is empty or the dimension of
   *the second operand is not a factor of that of the first operand.
   **/
  template <class OtherType>
  auto operator*(const Vector<OtherType> &other) const {
    if (IsEmpty())
      throw Exceptions::EmptyVector("Vector: Cannot perform element-wise "
                                    "multiplication on an empty vector.");
    if (other.Dimension() == 0)
      throw Exceptions::InvalidArgument(
          "Vector: Cannot perform element-wise multiplication with an empty "
          "vector as the second operand.");
    if (this->Dimension() % other.Dimension() != 0)
      throw Exceptions::InvalidArgument(
          "Vector: Expect the dimension of the second operand is a factor of "
          "that "
          "of the first operand when performing element-wise multiplication.");
    Vector<decltype((*this)[0] * other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Size(); i++)
      result[i] *= other[i % other.Dimension()];
    return result;
  }

  /**
   * Scale a Vector inplace.
   * @param scaler a scaler to be multiplied by each element of the Vector.
   * @return the reference to the Vector.
   * @throw EmptyVector when this vector is empty.
   **/
  template <class ScalerType> Vector<T> &operator*=(const ScalerType &scaler) {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform scaling on an empty vector.");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      (*this)[i] *= scaler;
    return *this;
  }

  /**
   * Perform inplace element-wise Vector multiplication.
   * @param other a Vector as the second operand.
   * @return the reference to this Vector.
   * @throw EmptyVector if this vector is empty.
   * @throw InvalidArgument if the dimensions of the Vectors are different.
   **/
  template <class OtherType>
  Vector<T> &operator*=(const Vector<OtherType> &other) {
    if (Dimension() == 0)
      throw Exceptions::EmptyVector("Vector: Cannot perform element-wise "
                                    "multiplication on an empty vector.");
    if (!other.Dimension())
      throw Exceptions::InvalidArgument(
          "Vector: Cannot perform element-wise multiplication with an empty "
          "vector as the second operand.");
    if (Dimension() % other.Dimension() != 0)
      throw Exceptions::InvalidArgument(
          "Vector: Expect the dimension of the second operand is a factor of "
          "that "
          "of the first operand when performing element-wise multiplication.");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      (*this)[i] *= other[i % other.Dimension()];
    return *this;
  }

  /**
   * Divide each element of the Vector by a scaler.
   * @param scaler a scaler to divide each element of the Vector.
   * @return a new Vector as the division result.
   * @throw EmptyVector if the Vector is empty.
   * @throw DividedByZero if the scaler is 0.
   **/
  template <class OtherType> auto Divide(const OtherType &scaler) const {
    if (Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform element-wise division on an empty vector.");
    else if (scaler == 0)
      throw Exceptions::DividedByZero("Vector: Cannot perform element-wise "
                                      "division as the second operand is 0.");
    Vector<decltype((*this)[0] / scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      result[i] /= scaler;
    return result;
  }

  /**
   * Perform element-wise division with two Vectors.
   * @param vector a Vector as the second operand.
   * @return a new Vector as the division result.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty of the dimension of
   *the first operand is not a factor of that of the second operand.
   * @throw DividedByZero if one of the elements of the second operand is 0.
   **/
  template <class OtherType>
  auto Divide(const Vector<OtherType> &vector) const {
    if (Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform element-wise division on an empty vector.");
    else if (vector.Dimension() == 0)
      throw Exceptions::InvalidArgument("Vector: Cannot perform element-wise "
                                        "division as the second operand is "
                                        "empty.");
    else if (Dimension() % vector.Dimension() != 0)
      throw Exceptions::InvalidArgument(
          "Vector: Cannot perform element-wise division. Expected the "
          "dimension "
          "of the "
          "second operand to be a factor of that of the first operand.");
    Vector<decltype((*this)[0] / vector[0])> result(*this);
    size_t j;

#pragma omp parallel for schedule(dynamic) private(j)
    for (size_t i = 0; i < Dimension(); i++) {
      j = i % vector.Dimension();
      if (vector[j] == 0)
        throw Exceptions::DividedByZero(
            "Vector: Expect none of the element of the second operand to be 0 "
            "when performing"
            "element-wise division.");
      result[i] /= vector[j];
    }
    return result;
  }

  /**
   * Divide each element of a Vector by a scaler. (See Vector::Divide)
   * @param scaler a scaler used to divide each element of the Vector.
   * @return a new Vector as the division result.
   * @throw EmptyVector if the Vector is empty.
   * @throw DividedByZero if the scaler is 0.
   **/
  template <class OtherType> auto operator/(const OtherType &scaler) const {
    return this->Divide(scaler);
  }

  /**
   * Perform element-wise division with two Vectors. (See Vector::Divide)
   * @param vector a Vector as the second operand.
   * @return a new Vector as the division result.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty of the dimension of
   * the first operand is not a factor of that of the second operand.
   * @throw DividedByZero if one of the elements of the second operand is 0.
   **/
  template <class OtherType>
  auto operator/(const Vector<OtherType> &vector) const {
    return this->Divide(vector);
  }

  /**
   * Perform inplace element-wise division with a Vector and a scaler.
   * @param scaler a scaler to divide each element of the Vector.
   * @return a reference to this Vector.
   * @throw EmptyVector if this Vector is empty.
   * @throw DividedByZero if the scaler is 0.
   **/
  template <class OtherType> Vector<T> &operator/=(const OtherType &scaler) {
    if (Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform element-wise division on an empty vector.");
    else if (scaler == 0)
      throw Exceptions::DividedByZero("Vector: Cannot perform element-wise "
                                      "division as the second operand is 0.");

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++)
      (*this)[i] /= scaler;
    return *this;
  }

  /**
   * Perform inplace element-wise division.
   * @param vector a Vector as the second operand.
   * @return a reference to this Vector.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty of the dimension of
   *the first operand is not a factor of that of the second operand.
   * @throw DividedByZero if one of the elements of the second operand is 0.
   **/
  template <class OtherType>
  Vector<T> &operator/=(const Vector<OtherType> &vector) {
    if (Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform element-wise division on an empty vector.");
    else if (vector.Dimension() == 0)
      throw Exceptions::InvalidArgument("Vector: Cannot perform element-wise "
                                        "division when the second operand "
                                        "is empty.");
    else if (Dimension() % vector.Dimension() != 0)
      throw Exceptions::InvalidArgument(
          "Vector: Cannot perform element-wise division. Expected the "
          "dimension "
          "of the "
          "second operand to be a factor of that of the first operand.");

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Dimension(); i++) {
      const auto element = vector[i % vector.Dimension()];
      if (element == 0)
        throw Exceptions::DividedByZero(
            "Vector: Expect none of the element of the second operand to be 0 "
            "when performing"
            "element-wise division.");
      (*this)[i] /= element;
    }
    return *this;
  }

  /**
   * Perform dot product with two Vectors.
   * @param other a Vector as the second operand.
   * @return a scaler as the dot product result.
   * @throw EmptyVector if this Vector is empty.
   * @throw InvalidArgument if the second operand is empty or the Vectors have
   *different dimensions.
   **/
  template <class OtherType> auto Dot(const Vector<OtherType> &other) const {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform dot product on an empty vector.");
    else if (other.IsEmpty())
      throw Exceptions::InvalidArgument("Vector: Cannot perform dot product "
                                        "when the second operand is empty.");
    else if (Dimension() != other.Dimension())
      throw Exceptions::InvalidArgument("Vector: Cannot perform dot product on "
                                        "vectors with different dimensions.");
    return Scale(other).Sum();
  }

  /**
   * Return the normalized Vector.
   * @return a new Vector with the normalized values.
   * @throw EmptyVector if the Vector is empty.
   * @throw DividedByZero if this Vector is a zero vector.
   **/
  template <class ReturnType> Vector<ReturnType> Normalized() const {
    if (Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform normalization on an empty vector.");
    const ReturnType length = Length<ReturnType>();
    if (length == 0)
      throw Exceptions::DividedByZero(
          "Vector: Cannot normalize a zero vector.");
    return *this / length;
  }

  /**
   * Normalize the Vector inplace.
   * @throw EmptyVector if the Vector is empty.
   * @throw DividedByZero if this Vector is a zero Vector.
   **/
  void Normalize() {
    if (IsEmpty())
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform normalization on an empty vector.");
    const T length = Length<T>();
    if (length == 0)
      throw Exceptions::DividedByZero(
          "Vector: Cannot normalize a zero vector.");
    *this /= length;
  }

  /**
   * Calculate the sum of all the elements of the Vector.
   * @return the sum of the elements.
   **/
  T Sum() const {
    T total = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : total)
    for (size_t i = 0; i < Size(); i++)
      total += (*this)[i];
    return total;
  }

  /**
   * Map each element of the Vector to a new value.
   * @param f a function that maps the value of an element to a new value.
   * @return a new Vector with the new values produced by f.
   **/
  template <class MapFunction> auto Map(MapFunction &&f) const {
    Vector<decltype(f((*this)[0]))> result(*this);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < Size(); i++)
      result[i] = f(result[i]);
    return result;
  }

  /**
   * Return a pointer that points to the first element of the Vector.
   * @return a pointer pointing to the first element of the Vector.
   **/
  const T *AsRawPointer() const { return this->data; }

  /**
   * Generate a Vector filled with zeros.
   * @param n the number of zeros.
   * @return a Vector that is filled with n zeros.
   **/
  static Vector<T> ZeroVector(const size_t &n) { return Vector<T>(n, 0); }

  /**
   * Stack all the elements of multiple Vectors in a new Vector.
   * @param vectors a std::initializer_list of Vectors to be stacked.
   * @return a Vector with the stacked elements.
   **/
  static Vector<T> Combine(const std::initializer_list<Vector<T>> &vectors) {
    size_t elementTotal = 0;
    for (auto itr = vectors.begin(); itr != vectors.end(); itr++) {
      elementTotal += itr->Size();
    }
    Vector<T> combined(elementTotal, 0);
    size_t currentIndex = 0;
    for (auto vector : vectors)
      for (size_t j = 0; j < vector.Size(); j++)
        combined[currentIndex++] = vector[j];
    return combined;
  }

  template <class ScalerType>
  friend auto operator+(const ScalerType &scaler, const Vector<T> &v) {
    Vector<decltype(scaler + v[0])> result(v);
    if (v.Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform addition on an empty vector.");
#pragma omp parallel for
    for (size_t i = 0; i < result.Dimension(); i++)
      result[i] += scaler;
    return result;
  }

  template <class ScalerType>
  friend auto operator-(const ScalerType &scaler, const Vector<T> &v) {
    Vector<decltype(scaler - v[0])> result(v);
    if (v.Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform subtraction on an empty vector.");
#pragma omp parallel for
    for (size_t i = 0; i < result.Dimension(); i++)
      result[i] = scaler - result[i];
    return result;
  }

  template <class ScalerType>
  friend auto operator*(const ScalerType &scaler, const Vector<T> &v) {
    Vector<decltype(scaler * v[0])> result(v);
    if (v.Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform scaling on an empty vector.");
#pragma omp parallel for
    for (size_t i = 0; i < result.Dimension(); i++)
      result[i] *= scaler;
    return result;
  }

  template <class ScalerType>
  friend auto operator/(const ScalerType &scaler, const Vector<T> &v) {
    Vector<decltype(scaler / v[0])> result(v);
    if (v.Dimension() == 0)
      throw Exceptions::EmptyVector(
          "Vector: Cannot perform element-wise division on an empty vector.");
    for (size_t i = 0; i < result.Dimension(); i++) {
      if (!result[i])
        throw Exceptions::DividedByZero(
            "Vector: Expect none of the elements of the second operand to be 0 "
            "when performing element-wise division.");
    }
#pragma omp parallel for
    for (size_t i = 0; i < result.Dimension(); i++)
      result[i] = scaler / result[i];
    return result;
  }

  template <class OtherType> friend class Vector;
};
} // namespace DataStructures

#endif
