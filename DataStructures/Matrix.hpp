#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <sstream>
#include <functional>
#include <type_traits>

#include "Container.hpp"
#include "Vector.hpp"
#include "List.hpp"
#include "Tuple.hpp"

namespace DataStructure
{
    /*
    Matrix class contains a collection of Vectors.
    */
    template <class T>
    class Matrix : public Container<Vector<T>>
    {
    private:
        // the number of row vectors
        std::size_t nRows;
        // the number of columns of each row vector.
        std::size_t nColumns;

    public:
        /*
        Constructor that creates an empty matrix.
        */
        Matrix();

        /*
        Constructor with the Number of Rows and Columns Specified.
        @param numRows the number of rows.
        @param numCoumns the number of columns.
        @param value the value, 0 by default, the Matrix will be filled with.
        */
        Matrix(std::size_t numRows, std::size_t numColumns, T initialValue = 0);

        /*
        Constructor with an initializer_list of row Vectors.
        @param l an initializer_list that contains row Vectors.
        @throw DimensionMismatch when there is any of the row Vectors has a
        different dimension.
        */
        Matrix(std::initializer_list<Vector<T>> l);

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the row vectors this Matrix will store.
        */
        template <std::size_t N>
        Matrix(const std::array<Vector<T>, N> &arr);

        /*
        Constructor with List as Input.
        @param l a List that contains the row vectors this Matrix will store.
        */
        Matrix(const List<Vector<T>> &l);

        /*
        Copy Constructor
        @param other a Matrix to be copied.
        */
        Matrix(const Matrix<T> &other);

        /*
        Copy Constructor
        @param other a Matrix to be copied.
        */
        template <class OtherType>
        Matrix(const Matrix<OtherType> &other);

        /*
        Move Constructor
        @param other a Matrix to be moved.
        */
        Matrix(Matrix<T> &&other);

        /*
        Move Constructor
        @param other a Matrix to be moved.
        */
        template <class OtherType>
        Matrix(Matrix<OtherType> &&other);

        /*
        Copy Assignment
        @param other a Matrix to be copied.
        @return a reference to this Matrix.
        */
        virtual Matrix<T> &operator=(const Matrix<T> &other);

        /*
        Accesses the vector at a given index.
        @return the vector at the given index.
        @throw IndexOutOfBound when the index exceeds the greatest possible index.
        */
        virtual Vector<T> &operator[](const std::size_t &index);

        /*
        Accesses the vector at a given index.
        @return the vector at the given index.
        @throw IndexOutOfBound when the index exceeds the greatest possible index.
        */
        virtual const Vector<T> &operator[](const std::size_t &index) const override;

        /*
        Returns the number of row Vectors this Matrix stores.
        @return the number of row Vectors this Matrix stores.
        */
        virtual std::size_t Size() const override;

        /*
        Returns the shape of this Matrix.
        @return a Tuple that contains the numbers of rows and columns.
        */
        Tuple<std::size_t> Shape() const;

        /*
        Performs matrix addition.
        @param other a Matrix.
        @return the result of the matrix addition.
        @throw InvalidArgument when the given matrix is empty.
        @throw EmptyMatrix when this matrix is empty.
        @throw MatrixShapeMismatch when the two shapes of the
        matrices do not match.
        */
        template <class OtherType>
        auto Add(const Matrix<OtherType> &other) const;

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
        auto operator+(const Matrix<OtherType> &other) const;

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
        Matrix<T> &operator+=(const Matrix<OtherType> &other);

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
        auto Subtract(const Matrix<OtherType> &other) const;

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
        auto operator-(const Matrix<OtherType> &other) const;

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
        Matrix<T> &operator-=(const Matrix<OtherType> &other);

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
        auto Multiply(const Matrix<OtherType> &other) const;

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
        auto operator*(const Matrix<OtherType> &other) const;

        /*
        Performs matrix scaling.
        @param scaler a scaler.
        @return the result of the matrix scaling.
        @throw EmptyMatrix when this matrix is empty.
        */
        auto Scale(const T &scaler) const;

        /*
        Performs Matrix element-wise multiplication.
        @param other a Matrix.
        @return the result of the matrix element-wise multiplication.
        @throw InvalidArgument when the target matrix is empty.
        @throw EmptyMatrix when this matrix is empty.
        @throw MatrixShapeMismatch when the shapes of the matrices
        are different.
        */
        template <class OtherType>
        auto Scale(const Matrix<OtherType> &other) const;

        /*
        Performs matrix scaling. Reference: Matrix.Scale
        @param scaler a scaler.
        @return the result of the matrix multiplication.
        @throw EmptyMatrix when this matrix is empty.
        */
        auto operator*(const T &scaler) const;

        /*
        Performs inplace matrix scaling.
        @param scaler a scaler.
        @return the reference to this Matrix.
        @throw EmptyMatrix when this matrix is empty.
        */
        Matrix<T> &operator*=(const T &scaler);

        /*
        Performs inplace matrix element-wise multiplication.
        @param other a scaler.
        @return the reference to this Matrix.
        @throw EmptyMatrix when this matrix is empty.
        */
        Matrix<T> &operator*=(const Matrix<T> &other);

        /*
        Performs matrix element-wise division.
        @param scaler a scaler.
        @return the result of the matrix element-wise division.
        @throw InvalidArgument when the scaler is zero.
        @throw EmptyMatrix when this matrix is empty.
        */
        template <class ScalerType>
        auto Divide(const ScalerType &scaler) const;

        /*
        Performs matrix element-wise division. Reference: Matrix.Divide.
        @param scaler a scaler.
        @return the result of the matrix element-wise division.
        @throw InvalidArgument when the scaler is zero.
        @throw EmptyMatrix when this matrix is empty.
        */
        template <class ScalerType>
        auto operator/(const ScalerType &scaler) const;

        /*
        Performs inplace matrix element-wise division.
        @param scaler a scaler.
        @return the reference to this Matrix.
        @throw InvalidArgument when the scaler is zero.
        @throw EmptyMatrix when this matrix is empty.
        */
        Matrix<T> &operator/=(const T &scaler);

        /*
        Converts this Matrix to a string that shows the elements of this Matrix.
        @return a string that represents this Matrix.
        */
        virtual std::string ToString() const override;

        friend std::ostream;

        /*
        Transposes this Matrix inplace.
        */
        void Transpose();

        /*
        Returns the transpose of this Matrix.
        @return the tranpose of this Matrix.
        */
        Matrix<T> Transposed() const;

        /*
        Constructs a new Matrix by flattening this Matrix in row-major or
        column-major order.
        @param rowMajor true if flattening in row-major. False if flattening
        in column-major order.
        @param keepInRow true if all the elements will be placed in a
        single row. False if they will be placed in a single column.
        @return a Matrix with a single row or column.
        */
        Matrix<T> Flattened(bool rowMajor = true, bool keepInRow = true) const;

        /*
        Calculate the summation of all the elements of this Matrix.
        @return the summatoin of the elements.
        */
        T Sum() const;

        /*
        Maps each element of this Matrix to a new value.
        @param f a function that maps the value of an element to a new value.
        @return a new Matrix with the new values defined by f.
        */
        template <class OtherType>
        Matrix<OtherType> Map(const std::function<OtherType(T)>& f) const;

        /*
        Constructs an identity matrix.
        @param n the size of the identity matrix to be generated.
        @return the identity matrix of size n.
        @throw InvalidArgument when n is non-positive.
        */
        static Matrix<T> Identity(std::size_t n);

        /*
        Constructs a diagonal matrix.
        @param values a Vector whose values will be the diagonal entries
        of the output Matrix.
        @param the diagonal matrix.
        */
        static Matrix<T> Diagonal(const Vector<T>& values);

        /*
        Constructs a translation matrix.
        @param deltas a translation vector.
        @return the translation matrix.
        */
        static Matrix<T> Translation(const Vector<T>& deltas);

        /*
        Constructs a scaling matrix.
        @param factors a vector having the factors on each axis.
        @return the scaling matrix defined by factors.
        */
        static Matrix<T> Scaling(const Vector<T>& factors);

        /*
        Constructs a rotation matrix in 2D space.
        @param radians the angle to rotate by.
        @return the rotation matrix defined by radians.
        */
        static Matrix<T> Rotation2D(const T& radians);

        /*
        Constructs a rotation matrix in 3D space.
        @param radians the angle to rotate by.
        @param axis a 3D vector that represents the axis to rotate around.
        @return the rotation matrix defined by radians.
        */
        static Matrix<T> Rotation3D(const Vector<T>& axis, const T& radians);

        /*
        Constructs a perspective projection matrix.
        @param fov the field of view in radians.
        @param aspect the aspect ratio (width / height of the viewport).
        @param near the distance from the camera to the near plane.
        @param far the distance from the camera to the far plane.
        */
        static Matrix<T> Perspective(T fov, T aspect, T near, T far);

        /*
        Constructs an othographic projection matrix.
        @param left the horizontal coordinate of the left of the frustum.
        @param right the horizontal coordinate of the right of the frustum.
        @param bottom the vertical coordinate of the bottom of the frustum.
        @param top the vertical coordinate of the top of the frustum.
        @param near the distance from the camera to the near plane.
        @param far the distance from the camera to the far plane.
        */
        static Matrix<T> Orthographic(T left, T right, T bottom, T top, T near, T far);

        template <class OtherType>
        friend class Matrix;
    };
} // namespace DataStructure

#include "Matrix.tpp"

#endif