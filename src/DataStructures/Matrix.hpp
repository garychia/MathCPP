#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <functional>
#include <sstream>
#include <type_traits>

#include "Container.hpp"
#include "List.hpp"
#include "Tuple.hpp"
#include "Vector.hpp"

namespace DataStructures {
/*
Matrix class contains a collection of Vectors.
*/
template <class T> class Matrix {
private:
  // the number of row vectors
  size_t nRows;
  // the number of columns of each row vector.
  size_t nColumns;
  // Elements of the matrix.
  Vector<T> elements;

public:
  /*
  Constructor that creates an empty matrix.
  */
  Matrix() : elements(), nRows(0), nColumns(0) {}

  /*
  Constructor with the Number of Rows and Columns Specified.
  @param numRows the number of rows.
  @param numCoumns the number of columns.
  @param value the value, 0 by default, the Matrix will be filled with.
  */
  Matrix(size_t numRows, size_t numColumns, const T &initialValue = 0)
      : elements(numRows * numColumns, initialValue), nRows(numRows),
        nColumns(numColumns) {}

  /*
  Constructor with an initializer_list of row Vectors.
  @param l an initializer_list that contains row Vectors.
  @throw DimensionMismatch when there is any of the row Vectors has a
  different dimension.
  */
  Matrix(std::initializer_list<Vector<T>> l)
      : elements(l.size() * l.begin()->Size()), nRows(l.size()),
        nColumns(l.begin()->Size()) {
    auto itr = l.begin();
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (size_t i = 0; i < nRows; i++) {
      for (size_t j = 0; j < nColumns; j++) {
        (*this)[i][j] = (*(itr + i))[j];
      }
    }
  }

  /*
  Constructor with an initializer_list of scalers. The resulting matrix
  will be a row or column vector matrix.
  @param l an initializer_list that contains the elements of the matrix.
  @param column true if the resulting matrix wiil be a column vector.
  Otherwise, a row vector will be constructed.
  */
  Matrix(std::initializer_list<T> l, bool column = true) : elements(l.size()) {
    nRows = column ? l.size() : 1;
    nColumns = column ? 1 : l.size();
    auto itr = l.begin();
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < l.size(); i++) {
      if (column)
        (*this)[i][0] = *(itr + i);
      else
        (*this)[0][i] = *(itr + i);
    }
  }

  /*
  Constructor with arrary as Input.
  @param arr an array that contains the row vectors this Matrix will store.
  */
  template <size_t N>
  Matrix(const std::array<Vector<T>, N> &arr)
      : elements(arr.size() * arr[0].Size()), nRows(arr.size()),
        nColumns(arr[0].Size()) {
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        (*this)[i][j] = arr[i][j];
      }
    }
  }

  /*
  Constructor with List as Input.
  @param l a List that contains the row vectors this Matrix will store.
  */
  Matrix(const List<Vector<T>> &l)
      : elements(l.Size() * l[0].Size()), nRows(l.Size()),
        nColumns(l[0].Size()) {
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (size_t i = 0; i < nRows; i++) {
      for (size_t j = 0; j < nColumns; j++) {
        (*this)[i][j] = l[i][j];
      }
    }
  }

  /*
  Copy Constructor
  @param other a Matrix to be copied.
  */
  Matrix(const Matrix<T> &other)
      : elements(other.elements), nRows(other.nRows), nColumns(other.nColumns) {}

  /*
  Copy Constructor
  @param other a Matrix to be copied.
  */
  template <class OtherType>
  Matrix(const Matrix<OtherType> &other)
      : elements(other.elements), nRows(other.nRows), nColumns(other.nColumns) {}

  /*
  Move Constructor
  @param other a Matrix to be moved.
  */
  Matrix(Matrix<T> &&other)
      : elements(std::move(other.elements)), nRows(other.nRows), nColumns(other.nColumns) {
    other.nRows = 0;
    other.nColumns = 0;
  }

  /*
  Move Constructor
  @param other a Matrix to be moved.
  */
  template <class OtherType>
  Matrix(Matrix<OtherType> &&other)
      : elements(std::move(other.elements)), nRows(other.nRows), nColumns(other.nColumns) {
    other.nRows = 0;
    other.nColumns = 0;
  }

  /*
  Copy Assignment
  @param other a Matrix to be copied.
  @return a reference to this Matrix.
  */
  virtual Matrix<T> &operator=(const Matrix<T> &other) {
    Matrix<T> temp(other);
    *this = std::move(temp);
    return *this;
  }

  virtual Matrix<T> &operator=(Matrix<T> &&other) {
    nRows = other.nRows;
    nColumns = other.nColumns;
    elements = std::move(other.elements);
    other.nRows = 0;
    other.nColumns = 0;
    return *this;
  }

  /*
  Accesses the vector at a given index.
  @return the vector at the given index.
  @throw IndexOutOfBound when the index exceeds the greatest possible index.
  */
  virtual T *operator[](const size_t &index) {
    return &elements[index * nColumns];
  }

  /*
  Accesses the vector at a given index.
  @return the vector at the given index.
  @throw IndexOutOfBound when the index exceeds the greatest possible index.
  */
  virtual const T *operator[](const size_t &index) const {
    return &elements[index * nColumns];
  }

  /*
  Returns the number of elements this Matrix stores.
  @return the number of elements this Matrix stores.
  */
  virtual size_t Size() const { return nRows * nColumns; }

  virtual bool IsEmpty() const { return elements.IsEmpty(); }

  /*
  Returns the shape of this Matrix.
  @return a Tuple that contains the numbers of rows and columns.
  */
  virtual Tuple<size_t> Shape() const {
    return Tuple<size_t>({nRows, nColumns});
  }

  /*
  Performs matrix addition.
  @param other a Matrix.
  @return the result of the matrix addition.
  @throw InvalidArgument when the given matrix is empty.
  @throw EmptyMatrix when this matrix is empty.
  @throw MatrixShapeMismatch when the two shapes of the
  matrices do not match.
  */
  template <class OtherType> auto Add(const Matrix<OtherType> &other) const {
    if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Matrix: Cannot perform addition with an empty matrix.");
    else if (this->IsEmpty())
      throw Exceptions::EmptyMatrix(
          "Matrix: Cannot perform addition with an empty matrix.");
    const auto otherShape = other.Shape();
    const auto thisShape = Shape();
    if (!(thisShape[0] % otherShape[0] == 0 &&
          thisShape[1] % otherShape[1] == 0)) {
      std::stringstream errorMessageStream;
      errorMessageStream << "Matrix: Expected the shape of the second operand "
                            "to be a factor of "
                            "the first one when performing matrix addition.";
      throw Exceptions::MatrixShapeMismatch(thisShape, otherShape,
                                            errorMessageStream.str());
    }

    Matrix<decltype((*this)[0][0] + other[0][0])> result(*this);
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        result[i][j] += other[i % other.nRows][j % other.nColumns];
      }
    }
    return result;
  }

  /*
  Performs matrix addition. Reference: Matrix.Add
  @param other a Matrix.
  @return the result of the matrix addition.
  @throw InvalidArgument when the given matrix is empty.
  @throw EmptyMatrix when this matrix is empty.
  @throw MatrixShapeMismatch when the shape of the first
  matrix is not a multiple of that of the second.
  */
  template <class OtherType>
  auto operator+(const Matrix<OtherType> &other) const {
    return Add(other);
  }

  /*
  Performs inplace matrix addition.
  @param other a Matrix.
  @return the reference to this Matrix.
  @throw InvalidArgument when the given matrix is empty.
  @throw EmptyMatrix when this matrix is empty.
  @throw MatrixShapeMismatch when the two shapes of the
  matrices are different.
  */
  template <class OtherType>
  Matrix<T> &operator+=(const Matrix<OtherType> &other) {
    if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Matrix: Cannot perform addition with an empty matrix.");
    else if (this->IsEmpty())
      throw Exceptions::EmptyMatrix(
          "Matrix: Cannot perform addition with an empty matrix.");
    const auto otherShape = other.Shape();
    const auto thisShape = Shape();
    if (!(thisShape[0] % otherShape[0] == 0 &&
          thisShape[1] % otherShape[1] == 0)) {
      std::stringstream errorMessageStream;
      errorMessageStream << "Matrix: Expected the shape of the second operand "
                            "to be a factor of "
                            "the first one when performing matrix addition.";
      throw Exceptions::MatrixShapeMismatch(thisShape, otherShape,
                                            errorMessageStream.str());
    }
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        (*this)[i][j] += other[i % other.nRows][j % other.nColumns];
      }
    }
    return result;
  }

  /*
  Performs matrix subtraction.
  @param other a Matrix.
  @return the result of the matrix subtraction.
  @throw InvalidArgument when the given matrix is empty.
  @throw EmptyMatrix when this matrix is empty.
  @throw MatrixShapeMismatch when the two shapes of the
  matrices are different.
  */
  template <class OtherType>
  auto Subtract(const Matrix<OtherType> &other) const {
    if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Matrix: Cannot perform subtraction with an empty matrix.");
    else if (this->IsEmpty())
      throw Exceptions::EmptyMatrix(
          "Matrix: Cannot perform subtraction with an empty matrix.");
    const auto otherShape = other.Shape();
    const auto thisShape = Shape();
    if (!(thisShape[0] % otherShape[0] == 0 &&
          thisShape[1] % otherShape[1] == 0)) {
      std::stringstream errorMessageStream;
      errorMessageStream
          << "Matrix: Expected the shape of the second operand to be a factor "
             "of that of the first operand when performing matrix subtraction."
          << std::endl;
      throw Exceptions::MatrixShapeMismatch(thisShape, otherShape,
                                            errorMessageStream.str());
    }

    Matrix<decltype((*this)[0][0] + other[0][0])> result(*this);
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        result[i][j] -= other[i % other.nRows][j % other.nColumns];
      }
    }
    return result;
  }

  /*
  Performs matrix subtraction. Reference: Matrix.Add
  @param other a Matrix.
  @return the result of the matrix subtraction.
  @throw InvalidArgument when the given matrix is empty.
  @throw EmptyMatrix when this matrix is empty.
  @throw MatrixShapeMismatch when the two shapes of the
  matrices are different.
  */
  template <class OtherType>
  auto operator-(const Matrix<OtherType> &other) const {
    return Subtract(other);
  }

  /*
  Performs inplace matrix subtraction.
  @param other a Matrix.
  @return the reference to this Matrix.
  @throw InvalidArgument when the given matrix is empty.
  @throw EmptyMatrix when this matrix is empty.
  @throw MatrixShapeMismatch when the two shapes of the
  matrices are different.
  */
  template <class OtherType>
  Matrix<T> &operator-=(const Matrix<OtherType> &other) {
    if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Matrix: Cannot perform subtraction with an empty matrix.");
    else if (this->IsEmpty())
      throw Exceptions::EmptyMatrix(
          "Matrix: Cannot perform subtraction with an empty matrix.");
    const auto otherShape = other.Shape();
    const auto thisShape = Shape();
    if (!(thisShape[0] % otherShape[0] == 0 &&
          thisShape[1] % otherShape[1] == 0)) {
      std::stringstream errorMessageStream;
      errorMessageStream << "Matrix: Expected the numbers of rows and columns "
                            "of the second matrix"
                            " to be factors of these of the first matrix."
                         << std::endl;
      throw Exceptions::MatrixShapeMismatch(thisShape, otherShape,
                                            errorMessageStream.str());
    }
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        (*this)[i][j] -= other[i % other.nRows][j % other.nColumns];
      }
    }
    return *this;
  }

  /*
  Performs matrix multiplication.
  @param other a Matrix.
  @return the result of the matrix multiplication.
  @throw InvalidArgument when the given matrix is empty.
  @throw EmptyMatrix when this matrix is empty.
  @throw MatrixShapeMismatch when the two shapes of the
  matrices do not match.
  */
  template <class OtherType>
  auto Multiply(const Matrix<OtherType> &other) const {
    if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Matrix: Cannot perform multiplication with an empty matrix.");
    else if (this->IsEmpty())
      throw Exceptions::EmptyMatrix(
          "Matrix: Cannot perform multiplication with an empty matrix.");
    const auto otherShape = other.Shape();
    const auto thisShape = Shape();
    if (thisShape[1] != otherShape[0]) {
      std::stringstream errorMessageStream;
      errorMessageStream
          << "Expected number of rows of the target matrix to be "
          << thisShape[1] << " when performing matrix multiplication.";
      throw Exceptions::MatrixShapeMismatch(thisShape, otherShape,
                                            errorMessageStream.str());
    }

    Matrix<decltype((*this)[0][0] * other[0][0])> result(thisShape[0],
                                                         otherShape[1]);
    size_t i, j, k;
#pragma omp parallel for private(k, j) schedule(dynamic) collapse(3)
    for (i = 0; i < thisShape[0]; i++)
      for (k = 0; k < thisShape[1]; k++)
        for (j = 0; j < otherShape[1]; j++)
#pragma omp atomic
          result[i][j] += (*this)[i][k] * other[k][j];
    return result;
  }

  /*
  Performs matrix multiplication. Reference: Matrix.Multiply
  @param other a Matrix.
  @return the result of the matrix multiplication.
  @throw InvalidArgument when the given matrix is empty.
  @throw EmptyMatrix when this matrix is empty.
  @throw MatrixShapeMismatch when the two shapes of the
  matrices do not match.
  */
  template <class OtherType>
  auto operator*(const Matrix<OtherType> &other) const {
    return Multiply(other);
  }

  /*
  Performs matrix scaling.
  @param scaler a scaler.
  @return the result of the matrix scaling.
  @throw EmptyMatrix when this matrix is empty.
  */
  auto Scale(const T &scaler) const {
    Matrix<decltype((*this)[0][0] * scaler)> result(*this);
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        result[i][j] *= scaler;
      }
    }
    return result;
  }

  /*
  Performs Matrix element-wise multiplication.
  @param other a Matrix.
  @return the result of the matrix element-wise multiplication.
  @throw InvalidArgument when the target matrix is empty.
  @throw EmptyMatrix when this matrix is empty.
  @throw MatrixShapeMismatch when the shapes of the matrices
  are different.
  */
  template <class OtherType> auto Scale(const Matrix<OtherType> &other) const {
    if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Matrix: Cannot perform element-wise scaling with an empty matrix.");
    if (this->IsEmpty())
      throw Exceptions::EmptyMatrix(
          "Matrix: Cannot perform element-wise scaling with an empty matrix.");

    const auto otherShape = other.Shape();
    const auto thisShape = Shape();
    if (thisShape[0] % otherShape[0] != 0 ||
        thisShape[1] % otherShape[1] != 0) {
      std::stringstream errorMessageStream;
      errorMessageStream
          << "Matrix: Expected the shape of the second operand to be a factor"
             " of the first one when performing matrix scaling.";
      throw Exceptions::MatrixShapeMismatch(thisShape, otherShape,
                                            errorMessageStream.str());
    }

    Matrix<decltype((*this)[0][0] * other[0][0])> result(*this);
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        result[i][j] *= other[i % other.nRows][j % other.nColumns];
      }
    }
    return result;
  }

  /*
  Performs matrix scaling. Reference: Matrix.Scale
  @param scaler a scaler.
  @return the result of the matrix multiplication.
  @throw EmptyMatrix when this matrix is empty.
  */
  auto operator*(const T &scaler) const { return Scale(scaler); }

  /*
  Performs inplace matrix scaling.
  @param scaler a scaler.
  @return the reference to this Matrix.
  @throw EmptyMatrix when this matrix is empty.
  */
  Matrix<T> &operator*=(const T &scaler) {
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        (*this)[i][j] *= scaler;
      }
    }
    return *this;
  }

  /*
  Performs inplace matrix element-wise multiplication.
  @param other a scaler.
  @return the reference to this Matrix.
  @throw EmptyMatrix when this matrix is empty.
  */
  Matrix<T> &operator*=(const Matrix<T> &other) {
    if (other.IsEmpty())
      throw Exceptions::InvalidArgument(
          "Matrix: Cannot perform element-wise scaling with an empty matrix.");
    if (this->IsEmpty())
      throw Exceptions::EmptyMatrix(
          "Matrix: Cannot perform element-wise scaling with an empty matrix.");

    const auto otherShape = other.Shape();
    const auto thisShape = Shape();
    if (thisShape[0] % otherShape[0] != 0 ||
        thisShape[1] % otherShape[1] != 0) {
      std::stringstream errorMessageStream;
      errorMessageStream
          << "Matrix: Expected the shape of the second operand to be a factor"
             " of the first one when performing matrix scaling.";
      throw Exceptions::MatrixShapeMismatch(thisShape, otherShape,
                                            errorMessageStream.str());
    }
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        (*this)[i][j] *= other[i % other.nRows][j % other.nColumns];
      }
    }
    return *this;
  }

  /*
  Performs matrix element-wise division.
  @param scaler a scaler.
  @return the result of the matrix element-wise division.
  @throw DividedByZero when the scaler is zero.
  */
  template <class ScalerType> auto Divide(const ScalerType &scaler) const {
    if (scaler == 0)
      throw Exceptions::DividedByZero(
          "Matrix: Cannot perform element-wise division with zero.");

    Matrix<decltype((*this)[0][0] * scaler)> result(*this);
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        result[i][j] /= scaler;
      }
    }
    return result;
  }

  /*
  Broadcasts elements of a smaller matrix and performs element-wise division.
  @param matrix another matrix.
  @return the result of the matrix division.
  @throw DividedByZero when there is a zero denominator.
  @throw EmptyMatrix when this matrix is empty.
  @throw InvalidArgument when broadcasting cannot be performed.
  */
  template <class OtherType>
  auto Divide(const Matrix<OtherType> &other) const {
    if (IsEmpty() || other.IsEmpty())
      throw Exceptions::EmptyMatrix(
          "Matrix: Cannot perform element-wise division on an empty matrix.");
    if (nRows % other.nRows != 0 || nColumns % other.nColumns != 0)
      throw Exceptions::InvalidArgument(
          "Matrix: The shape of the denominator matrix must be a factor of "
          "that of the numerator matrix.");

    Matrix<decltype((*this)[0][0] / other[0][0])> result(*this);
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        if (!other[i % other.nRows][j % other.nColumns]) {
          throw Exceptions::DividedByZero(
              "Matrix: Cannot perform element-wise division as there is a "
              "zero denominator.");
        }
        result[i][j] /= other[i % other.nRows][j % other.nColumns];
      }
    }
    return result;
  }

  /*
  Performs matrix element-wise division. Reference: Matrix.Divide.
  @param scaler a scaler.
  @return the result of the matrix element-wise division.
  @throw InvalidArgument when the scaler is zero.
  @throw EmptyMatrix when this matrix is empty.
  */
  template <class ScalerType> auto operator/(const ScalerType &scaler) const {
    return Divide(scaler);
  }

  /*
  Performs inplace matrix element-wise division.
  @param scaler a scaler.
  @return the reference to this Matrix.
  @throw InvalidArgument when the scaler is zero.
  */
  Matrix<T> &operator/=(const T &scaler) {
    if (scaler == 0)
      throw Exceptions::InvalidArgument(
          "Matrix: Cannot perform element-wise division with zero.");

    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = 0; j < nColumns; j++) {
        (*this)[i][j] /= scaler;
      }
    }
    return *this;
  }

  /*
  Converts this Matrix to a string that shows the elements of this Matrix.
  @return a string that represents this Matrix.
  */
  virtual std::string ToString() const {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < nRows; i++) {
      ss << (i == 0 ? "[" : " [");
      for (size_t j = 0; j < nColumns; j++) {
        ss << (*this)[i][j];
        if (j < nColumns - 1)
          ss << ", ";
      }
      ss << "]";
      if (i < nRows - 1)
        ss << "," << std::endl;
    }
    ss << "]";
    return ss.str();
  }

  friend std::ostream;

  /*
  Transposes this Matrix inplace.
  */
  void Transpose() {
    size_t i, j;
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
    for (i = 0; i < nRows; i++) {
      for (j = i + 1; j < nColumns; j++) {
        std::swap(elements[i * nColumns + j], elements[j * nRows + i]);
      }
    }
    std::swap(nRows, nColumns);
  }

  /*
  Returns the transpose of this Matrix.
  @return the tranpose of this Matrix.
  */
  Matrix<T> Transposed() const {
    Matrix<T> temp(*this);
    temp.Transpose();
    return temp;
  }

  /*
  Constructs a new Matrix by flattening this Matrix in row-major or
  column-major order.
  @param rowMajor true if flattening in row-major. False if flattening
  in column-major order.
  @param keepInRow true if all the elements will be placed in a
  single row. False if they will be placed in a single column.
  @return a Matrix with a single row or column.
  */
  Matrix<T> Flattened(bool rowMajor = true, bool keepInRow = true) const {
    Matrix<T> result(*this);
    if (!rowMajor)
      result.Transpose();
    const auto nElements = result.nRows * result.nColumns;
    result.nColumns = keepInRow ? nElements : 1;
    result.nRows = keepInRow ? 1 : nElements;
    return result;
  }

  /*
   *Calculate the summation of all the elements of this Matrix.
   *@return the summation of the elements.
   */
  T SumAll() const {
      return elements.Sum();
  }

  /*
  Calculate the summation of all the rows or columns of this Matrix.
  @param sumRows true then the summation of rows will be returned.
  Otherwise, the summation of columns will be returned instead.
  @return the summation of the rows or columns.
  */
  Matrix<T> Sum(bool sumRows = true) const {
    Matrix<T> result(sumRows ? 1 : nRows, sumRows ? nColumns : 1);
    size_t i, j;
    if (sumRows) {
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
      for (size_t i = 0; i < nRows; i++) {
        for (size_t j = 0; j < nColumns; j++) {
          result[0][j] += (*this)[i][j];
        }
      }
    } else {
#pragma omp parallel for private(j) collapse(2) schedule(dynamic)
      for (size_t i = 0; i < nRows; i++) {
        for (size_t j = 0; j < nColumns; j++) {
          result[i][0] += (*this)[i][j];
        }
      }
    }
    return result;
  }

  /*
  Calculates the Frobenius norm of a Matrix.
  @param m a Matrix.
  @return the Frobenius norm of the given Matrix.
  */
  template <class ReturnType> ReturnType FrobeniusNorm() const {
    return Math::Power<T, double>(
        Map([](const T &e) { return e * e; }).SumAll(), 0.5);
  }

  /*
  Maps each element of this Matrix to a new value.
  @param f a function that maps the value of an element to a new value.
  @return a new Matrix with the new values defined by f.
  */
  template <class MapFunction> auto Map(MapFunction &&f) const {
    Matrix<T> result(*this);
    result.elements = result.elements.Map(std::forward<MapFunction>(f));
    return result;
  }

  /*
  Constructs an identity matrix.
  @param n the size of the identity matrix to be generated.
  @return the identity matrix of size n.
  @throw InvalidArgument when n is non-positive.
  */
  static Matrix<T> Identity(size_t n) {
    Matrix<T> result(n, n);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; i++)
      result[i][i] = 1;
    return result;
  }

  /*
  Constructs a diagonal matrix.
  @param values a Vector whose values will be the diagonal entries
  of the output Matrix.
  @param the diagonal matrix.
  */
  static Matrix<T> Diagonal(const Vector<T> &values) {
    const auto n = values.Dimension();
    auto result = Identity(n, n);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; i++)
      result[i][i] = values[i];
    return result;
  }

  /*
  Constructs a translation matrix.
  @param deltas a translation vector.
  @return the translation matrix.
  */
  static Matrix<T> Translation(const Vector<T>& deltas) {
    const size_t n = deltas.Dimension() + 1;
    auto result = Identity(n);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n - 1; i++)
      result[i][n - 1] = deltas[i];
    return result;
  }

  /*
  Constructs a scaling matrix.
  @param factors a vector having the factors on each axis.
  @return the scaling matrix defined by factors.
  */
  static Matrix<T> Scaling(const Vector<T>& factors) {
    const size_t n = factors.Dimension() + 1;
    auto result = Identity(n);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n - 1; i++)
      result[i][i] *= factors[i];
    return result;
  }

  /*
  Constructs a rotation matrix in 2D space.
  @param radians the angle to rotate by.
  @return the rotation matrix defined by radians.
  */
  static Matrix<T> Rotation2D(const T &radians) {
    const T sinValue = Math::Sine(radians);
    const T cosValue = Math::Cosine(radians);
    return Matrix<T>(
        {{cosValue, -sinValue, 0}, {sinValue, cosValue, 0}, {0, 0, 1}});
  }

  /*
  Constructs a rotation matrix in 3D space.
  @param radians the angle to rotate by.
  @param axis a 3D vector that represents the axis to rotate around.
  @return the rotation matrix defined by radians.
  */
  static Matrix<T> Rotation3D(const Vector<T>& axis, const T& radians) {
    if (axis.Dimension() != 3) {
      std::stringstream ss;
      ss << "Matrix: Rotation3D requires an axis defined in 3D space "
            "but got an axis of dimension "
         << axis.Dimension() << ".";
      throw Exceptions::InvalidArgument(ss.str());
    }
    const auto normalizedAxis = axis.Normalized();
    const T &x = normalizedAxis[0];
    const T &y = normalizedAxis[1];
    const T &z = normalizedAxis[2];
    const T sinValue = Math::Sine(radians);
    const T cosValue = Math::Cosine(radians);
    const T oneMinusCosValue = 1 - cosValue;
    return Matrix<T>({{cosValue + x * x * oneMinusCosValue,
                       x * y * oneMinusCosValue - z * sinValue,
                       x * z * oneMinusCosValue + y * sinValue, 0},
                      {y * x * oneMinusCosValue + z * sinValue,
                       cosValue + y * y * oneMinusCosValue,
                       y * z * oneMinusCosValue - x * sinValue, 0},
                      {z * x * oneMinusCosValue - y * sinValue,
                       z * y * oneMinusCosValue + x * sinValue,
                       cosValue + z * z * oneMinusCosValue, 0},
                      {0, 0, 0, 1}});
  }

  /*
  Constructs a perspective projection matrix.
  @param fov the field of view in radians.
  @param aspect the aspect ratio (width / height of the viewport).
  @param near the distance from the camera to the near plane.
  @param far the distance from the camera to the far plane.
  */
  static Matrix<T> Perspective(T fov, T aspect, T near, T far) {
    if (fov <= 0) {
      std::stringstream ss;
      ss << "Matrix::Perspective: Field of View must be positive but got "
         << fov;
      throw Exceptions::InvalidArgument(ss.str());
    }
    if (aspect == 0) {
      std::stringstream ss;
      ss << "Matrix::Perspective: Aspect Ratio must be non-zero but got "
         << aspect;
      throw Exceptions::InvalidArgument(ss.str());
    }
    const T scale = 1 / Math::Tangent(fov * 0.5);
    const T farNearDiff = far - near;
    return Matrix<T>(
        {{scale * aspect, 0, 0, 0},
         {0, scale, 0, 0},
         {0, 0, -(far + near) / farNearDiff, -2 * near * far / farNearDiff},
         {0, 0, -1, 0}});
  }

  /*
  Constructs an othographic projection matrix.
  @param left the horizontal coordinate of the left of the frustum.
  @param right the horizontal coordinate of the right of the frustum.
  @param bottom the vertical coordinate of the bottom of the frustum.
  @param top the vertical coordinate of the top of the frustum.
  @param near the distance from the camera to the near plane.
  @param far the distance from the camera to the far plane.
  */
  static Matrix<T> Orthographic(T left, T right, T bottom, T top, T near,
                                T far) {
    if (left == right)
      throw Exceptions::InvalidArgument(
          "Matrix::Orthographic: 'left' and 'right' should not be the same "
          "value.");
    if (bottom == top)
      throw Exceptions::InvalidArgument(
          "Matrix::Orthographic: 'top' and 'bottom' should not be the same "
          "value.");
    if (near == far)
      throw Exceptions::InvalidArgument("Matrix::Orthographic: 'near' and "
                                        "'far' should not be the same value.");
    const T rightLeftDiff = right - left;
    const T topBottomDiff = top - bottom;
    const T farNearDist = far - near;
    return Matrix<T>(
        {{2 / rightLeftDiff, 0, 0, -(right + left) / rightLeftDiff},
         {0, 2 / topBottomDiff, 0, -(top + bottom) / topBottomDiff},
         {0, 0, -2 / farNearDist, -(far + near) / farNearDist},
         {0, 0, 0, 1}});
  }

  /**
   * Operator * with a scaler and a Matrix.
   * @param scaler a scaler.
   * @param matrix a Matrix.
   * @return a Matrix with the elements of 'matrix' multiplied by the scaler.
   **/
  friend auto operator*(const double &scaler, const Matrix<T> &matrix) {
    return matrix.Map([&scaler](T e) { return scaler * e; });
  }

  /**
   * Operator / with a scaler and a Matrix.
   * @param scaler a scaler.
   * @param matrix a Matrix.
   * @return a Matrix with the scaler divided by each element of 'matrix'.
   **/
  friend auto operator/(const double &scaler, const Matrix<T> &matrix) {
    return matrix.Map([&scaler](T e) { return scaler / e; });
  }

  template <class OtherType> friend class Matrix;
};

/**
 *Converts this Matrix to a string and pass it to a output stream.
 *@param os an output stream.
 *@param m a Matrix
 *@return the output stream.
 */
template <class T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &m) {
  os << m.ToString();
  return os;
}

} // namespace DataStructures

#endif
