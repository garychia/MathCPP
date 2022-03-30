#ifndef MATRIX_H
#define MATRIX_H

#include <sstream>

#include "container.hpp"
#include "vector.hpp"
#include "list.hpp"
#include "../Exceptions/exceptions.hpp"

namespace DataStructure
{
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
        Matrix() : Container<Vector<T>>(), nRows(0), nColumns(0) {}

        /*
        Constructor with the Number of Rows and Columns Specified.
        @param numRows the number of rows.
        @param numCoumns the number of columns.
        @param value the value, 0 by default, the Matrix will be filled with.
        */
        Matrix(std::size_t numRows, std::size_t numColumns, T initialValue = 0)
        {
            nRows = numRows;
            nColumns = numColumns;
            this->size = numRows;
            this->data = nullptr;
            if (numRows > 0)
            {
                if (numColumns == 0)
                    throw Exceptions::InvalidArgument(
                        "Matrix: the number of columns must be a positive number for a non-empty matrix.");

                this->data = new Vector<T>[numRows];

                std::size_t i;
#pragma omp parallel for schedule(dynamic)
                for (i = 0; i < numRows; i++)
                    this->data[i] = Vector<T>::ZeroVector(numColumns);

                if (initialValue != 0)
                {
                    std::size_t j;
#pragma omp parallel for private(j) schedule(dynamic) collapse(2)
                    for (i = 0; i < numRows; i++)
                        for (j = 0; j < numColumns; j++)
                            this->data[i][j] = initialValue;
                }
            }
            else if (numColumns > 0)
            {
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot initialize a matrix with no row vector while the number of columns is greater than 0.");
            }
        }

        /*
        Constructor with an initializer_list of row Vectors.
        @param l an initializer_list that contains row Vectors.
        @throw DimensionMismatch when there is any of the row Vectors has a
        different dimension.
        */
        Matrix(std::initializer_list<Vector<T>> l) : Container<Vector<T>>(l)
        {
            if (l.size() > 0)
            {
                nRows = l.size();
                nColumns = l.begin()->Size();
                for (auto itr = l.begin() + 1; itr != l.end(); itr++)
                {
                    if (itr->Dimension() != nColumns)
                        throw Exceptions::DimensionMismatch(
                            nColumns,
                            itr->Dimension(),
                            "Matrix: All row vectors must be of the same dimension.");
                }
            }
            else
            {
                nRows = 0;
                nColumns = 0;
                this->size = 0;
                this->data = nullptr;
            }
        }

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the row vectors this Matrix will store.
        */
        template <std::size_t N>
        Matrix(const std::array<Vector<T>, N> &arr)
        {
            if (arr.size() > 0)
            {
                nRows = arr.size();
                nColumns = arr.begin()->Size();
                this->size = nRows;
                for (std::size_t i = 0; i < nRows; i++)
                {
                    if (arr[i].Dimension() != nColumns)
                        throw Exceptions::DimensionMismatch(
                            nColumns,
                            arr[i].Dimension(),
                            "Matrix: All row vectors must be of the same dimension.");
                }
                Container<Vector<T>>::Container(arr);
            }
            else
            {
                nRows = 0;
                nColumns = 0;
                this->size = 0;
                this->data = nullptr;
            }
        }

        /*
        Constructor with List as Input.
        @param l a List that contains the row vectors this Matrix will store.
        */
        Matrix(const List<Vector<T>> &l)
        {
            const std::size_t listSize = l.Size();
            if (listSize > 0)
            {
                nRows = listSize;
                nColumns = l[0].Size();
                this->size = nRows;
#pragma omp parallel for schedule(dynamic)
                for (std::size_t i = 0; i < nRows; i++)
                {
                    if (l[i].Dimension() != nColumns)
                        throw Exceptions::DimensionMismatch(
                            nColumns,
                            l[i].Dimension(),
                            "Matrix: All row vectors must be of the same dimension.");
                }
                this->data = new Vector<T>[nRows];
#pragma omp parallel for schedule(dynamic)
                for (std::size_t i = 0; i < nRows; i++)
                    this->data[i] = l[i];
            }
            else
            {
                nRows = 0;
                nColumns = 0;
                this->size = 0;
                this->data = nullptr;
            }
        }

        /*
        Copy Constructor
        @param other a Matrix to be copied.
        */
        Matrix(const Matrix<T> &other) : Container<Vector<T>>(other)
        {
            nRows = other.nRows;
            nColumns = (nRows > 0) ? this->data[0].Size() : 0;
        }

        /*
        Copy Constructor
        @param other a Matrix to be copied.
        */
        template <class OtherType>
        Matrix(const Matrix<OtherType> &other) : Container<Vector<T>>(other)
        {
            nRows = other.Shape()[0];
            nColumns = (nRows > 0) ? this->data[0].Size() : 0;
        }

        /*
        Move Constructor
        @param other a Matrix to be moved.
        */
        Matrix(Matrix<T> &&other) : Container<Vector<T>>(other)
        {
            nRows = std::move(other.nRows);
            nColumns = (nRows > 0) ? (*this)[0].Size() : 0;
            other.nRows = 0;
            other.nColumns = 0;
        }

        /*
        Move Constructor
        @param other a Matrix to be moved.
        */
        template <class OtherType>
        Matrix(Matrix<OtherType> &&other) : Container<Vector<T>>(other)
        {
            nRows = std::move(other.Shape()[0]);
            nColumns = (nRows > 0) ? (*this)[0].Size() : 0;
            other.nRows = 0;
            other.nColumns = 0;
        }

        /*
        Accesses the vector at a given index.
        @return the vector at the given index.
        @throw IndexOutOfBound when the index exceeds the greatest possible index.
        */
        virtual Vector<T> &operator[](const std::size_t &index)
        {
            try
            {
                return this->data[index];
            }
            catch (const Exceptions::IndexOutOfBound &)
            {
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Matrix: Index must be less than the number of rows.");
            }
        }

        /*
        Accesses the vector at a given index.
        @return the vector at the given index.
        @throw IndexOutOfBound when the index exceeds the greatest possible index.
        */
        virtual const Vector<T> &operator[](const std::size_t &index) const override
        {
            try
            {
                return this->data[index];
            }
            catch (const Exceptions::IndexOutOfBound &)
            {
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Matrix: Index must be less than the number of rows.");
            }
        }

        /*
        Returns the number of row Vectors this Matrix stores.
        @return the number of row Vectors this Matrix stores.
        */
        virtual std::size_t Size() const override { return this->size; }

        /*
        Returns the shape of this Matrix.
        @return a Tuple that contains the numbers of rows and columns.
        */
        Tuple<std::size_t> Shape() const { return Tuple<std::size_t>({nRows, nColumns}); }

        /*
        Performs matrix addition.
        @param other a Matrix.
        @return the result of the matrix addition.
        @throw InvalidArgument when the given matrix is empty.
        @throw EmptyMatrix when this matrix is empty.
        @throw MatrixShapeMismatch when the two shapes of the
        matrices are different.
        */
        template <class OtherType>
        auto Add(const Matrix<OtherType> &other) const
        {
            if (other.IsEmpty())
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot perform addition with an empty matrix.");
            else if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform addition with an empty matrix.");
            const auto otherShape = other.Shape();
            const auto thisShape = Shape();
            if (thisShape[1] != otherShape[0])
            {
                std::stringstream errorMessageStream;
                errorMessageStream
                    << "Expected shape of the target matrix to be "
                    << thisShape
                    << "when performing addition."
                    << std::endl;
                throw Exceptions::MatrixShapeMismatch(
                    thisShape,
                    otherShape,
                    errorMessageStream.str());
            }

            Matrix<decltype((*this)[0][0] + other[0][0])> result(*this);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                result[i] += other[i];
            return result;
        }

        /*
        Performs matrix addition. Reference: Matrix.Add
        @param other a Matrix.
        @return the result of the matrix addition.
        @throw InvalidArgument when the given matrix is empty.
        @throw EmptyMatrix when this matrix is empty.
        @throw MatrixShapeMismatch when the two shapes of the
        matrices are different.
        */
        template <class OtherType>
        auto operator+(const Matrix<OtherType> &other) const
        {
            try
            {
                return Add(other);
            }
            catch (Exceptions::InvalidArgument &e)
            {
                throw e;
            }
            catch (Exceptions::EmptyMatrix &e)
            {
                throw e;
            }
            catch (Exceptions::MatrixShapeMismatch &e)
            {
                throw e;
            }
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
        Matrix<T> &operator+=(const Matrix<OtherType> &other)
        {
            if (other.IsEmpty())
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot perform addition with an empty matrix.");
            else if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform addition with an empty matrix.");
            const auto otherShape = other.Shape();
            const auto thisShape = Shape();
            if (thisShape[1] != otherShape[0])
            {
                std::stringstream errorMessageStream;
                errorMessageStream
                    << "Expected shape of the target matrix to be "
                    << thisShape
                    << "when performing addition."
                    << std::endl;
                throw Exceptions::MatrixShapeMismatch(
                    thisShape,
                    otherShape,
                    errorMessageStream.str());
            }

#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                (*this)[i] += other[i];
            return *this;
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
        auto Subtract(const Matrix<OtherType> &other) const
        {
            if (other.IsEmpty())
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot perform subtraction with an empty matrix.");
            else if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform subtraction with an empty matrix.");
            const auto otherShape = other.Shape();
            const auto thisShape = Shape();
            if (thisShape[1] != otherShape[0])
            {
                std::stringstream errorMessageStream;
                errorMessageStream
                    << "Expected shape of the target matrix to be "
                    << thisShape
                    << "when performing subtraction."
                    << std::endl;
                throw Exceptions::MatrixShapeMismatch(
                    thisShape,
                    otherShape,
                    errorMessageStream.str());
            }

            Matrix<decltype((*this)[0][0] + other[0][0])> result(*this);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                result[i] -= other[i];
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
        auto operator-(const Matrix<OtherType> &other) const
        {
            try
            {
                return Subtract(other);
            }
            catch (Exceptions::InvalidArgument &e)
            {
                throw e;
            }
            catch (Exceptions::EmptyMatrix &e)
            {
                throw e;
            }
            catch (Exceptions::MatrixShapeMismatch &e)
            {
                throw e;
            }
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
        Matrix<T> &operator-=(const Matrix<OtherType> &other)
        {
            if (other.IsEmpty())
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot perform subtraction with an empty matrix.");
            else if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform subtraction with an empty matrix.");
            const auto otherShape = other.Shape();
            const auto thisShape = Shape();
            if (thisShape[1] != otherShape[0])
            {
                std::stringstream errorMessageStream;
                errorMessageStream
                    << "Expected shape of the target matrix to be "
                    << thisShape
                    << "when performing subtraction."
                    << std::endl;
                throw Exceptions::MatrixShapeMismatch(
                    thisShape,
                    otherShape,
                    errorMessageStream.str());
            }

#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                (*this)[i] -= other[i];
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
        auto Multiply(const Matrix<OtherType> &other) const
        {
            if (other.IsEmpty())
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot perform multiplication with an empty matrix.");
            else if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform multiplication with an empty matrix.");
            const auto otherShape = other.Shape();
            const auto thisShape = Shape();
            if (thisShape[1] != otherShape[0])
            {
                std::stringstream errorMessageStream;
                errorMessageStream
                    << "Expected number of rows of the target matrix to be "
                    << thisShape[1]
                    << "."
                    << std::endl;
                throw Exceptions::MatrixShapeMismatch(
                    thisShape,
                    otherShape,
                    errorMessageStream.str());
            }

            Matrix<decltype((*this)[0][0] * other[0][0])>
                result(thisShape[0], otherShape[1]);
            std::size_t i, j, k;

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
        auto operator*(const Matrix<OtherType> &other) const
        {
            try
            {
                return Multiply(other);
            }
            catch (Exceptions::InvalidArgument &e)
            {
                throw e;
            }
            catch (Exceptions::EmptyMatrix &e)
            {
                throw e;
            }
            catch (Exceptions::MatrixShapeMismatch &e)
            {
                throw e;
            }
        }

        /*
        Performs matrix scaling.
        @param scaler a scaler.
        @return the result of the matrix scaling.
        @throw EmptyMatrix when this matrix is empty.
        */
        auto Scale(const T &scaler) const
        {
            if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform scaling on an empty matrix.");

            Matrix<decltype((*this)[0][0] * scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                result[i] *= scaler;
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
        template <class OtherType>
        auto Scale(const Matrix<OtherType> &other) const
        {
            if (other.IsEmpty())
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot perform element-wise scaling with an empty matrix.");
            if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform element-wise scaling with an empty matrix.");

            const auto otherShape = other.Shape();
            const auto thisShape = Shape();
            if (thisShape != otherShape)
            {
                std::stringstream errorMessageStream;
                errorMessageStream
                    << "Expected two matrices have the same shape when performing"
                       "element-wise scaling."
                    << std::endl;
                throw Exceptions::MatrixShapeMismatch(
                    thisShape,
                    otherShape,
                    errorMessageStream.str());
            }

            Matrix<decltype((*this)[0][0] * other[0][0])> result(*this);
            #pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                result[i] *= other[i];
            return result;
        }

        /*
        Performs matrix scaling. Reference: Matrix.Scale
        @param scaler a scaler.
        @return the result of the matrix multiplication.
        @throw EmptyMatrix when this matrix is empty.
        */
        auto operator*(const T &scaler) const
        {
            try
            {
                return Scale(scaler);
            }
            catch (Exceptions::EmptyMatrix &e)
            {
                throw e;
            }
        }

        /*
        Performs inplace matrix scaling.
        @param scaler a scaler.
        @return the reference to this Matrix.
        @throw EmptyMatrix when this matrix is empty.
        */
        Matrix<T> &operator*=(const T &scaler)
        {
            if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform scaling on an empty matrix.");

#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                (*this)[i] *= scaler;
            return *this;
        }

        /*
        Performs inplace matrix element-wise multiplication.
        @param other a scaler.
        @return the reference to this Matrix.
        @throw EmptyMatrix when this matrix is empty.
        */
        Matrix<T> &operator*=(const Matrix<T> &other)
        {
            if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform scaling on an empty matrix.");

            #pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                (*this)[i] *= other[i];
            return *this;
        }

        /*
        Performs matrix element-wise division.
        @param scaler a scaler.
        @return the result of the matrix element-wise division.
        @throw InvalidArgument when the scaler is zero.
        @throw EmptyMatrix when this matrix is empty.
        */
        template <class ScalerType>
        auto Divide(const ScalerType &scaler) const
        {
            if (scaler == 0)
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot perform element-wise division with zero.");
            if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform element-wise division on an empty matrix.");

            Matrix<decltype((*this)[0][0] * scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                result[i] /= scaler;
            return result;
        }

        /*
        Performs matrix element-wise division. Reference: Matrix.Divide.
        @param scaler a scaler.
        @return the result of the matrix element-wise division.
        @throw InvalidArgument when the scaler is zero.
        @throw EmptyMatrix when this matrix is empty.
        */
        template <class ScalerType>
        auto operator/(const ScalerType &scaler) const
        {
            try
            {
                return Divide(scaler);
            }
            catch (Exceptions::InvalidArgument &e)
            {
                throw e;
            }
            catch (Exceptions::EmptyMatrix &e)
            {
                throw e;
            }
        }

        /*
        Performs inplace matrix element-wise division.
        @param scaler a scaler.
        @return the reference to this Matrix.
        @throw InvalidArgument when the scaler is zero.
        @throw EmptyMatrix when this matrix is empty.
        */
        Matrix<T> &operator/=(const T &scaler)
        {
            if (scaler == 0)
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot perform element-wise division with zero.");
            if (this->IsEmpty())
                throw Exceptions::EmptyMatrix(
                    "Matrix: Cannot perform scaling on an empty matrix.");

#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < nRows; i++)
                (*this)[i] /= scaler;
            return *this;
        }

        /*
        Converts this Matrix to a string that shows the elements of this Matrix.
        @return a string that represents this Matrix.
        */
        virtual std::string ToString() const override
        {
            if (nRows == 0)
                return "[EMPTY MATRIX]";
            std::stringstream ss;
            ss << "[";
            for (std::size_t i = 0; i < nRows; i++)
            {
                ss << (i == 0 ? "[" : " [");
                for (std::size_t j = 0; j < nColumns; j++)
                {
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

        /*
        Converts this Matrix to a string and pass it to a output stream.
        @param os an output stream.
        @param m a Matrix
        @return the output stream.
        */
        friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &m)
        {
            os << m.ToString();
            return os;
        }

        /*
        Transposes this Matrix inplace.
        */
        void Transpose()
        {
            if (nRows > 0)
            {
                std::size_t i, j;
                Vector<T> *newRows = new Vector<T>[nColumns];
#pragma omp parallel for schedule(dynamic)
                for (i = 0; i < nColumns; i++)
                    newRows[i] = Vector<T>::ZeroVector(nRows);
#pragma omp parallel for private(j) schedule(dynamic) collapse(2)
                for (i = 0; i < nColumns; i++)
                    for (j = 0; j < nRows; j++)
                        newRows[i][j] = this->data[j][i];
                delete[] this->data;
                this->data = newRows;
                std::swap(nRows, nColumns);
            }
        }

        /*
        Constructs an identity matrix.
        @param n the size of the identity matrix to be generated.
        @return the identity matrix of size n.
        @throw InvalidArgument when n is non-positive.
        */
        static Matrix<T> Identity(std::size_t n)
        {
            if (n == 0)
                throw Exceptions::InvalidArgument(
                    "The size of the identity matrix must be positive.");
            Matrix<T> identity(n, n);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < n; i++)
                identity[i][i] = 1;
            return identity;
        }


        /*
        Constructs a diagonal matrix.
        @param values a Vector whose values will be the diagonal entries
        of the output Matrix.
        @param the diagonal matrix.
        */
        static Matrix<T> Diagonal(const Vector<T>& values)
        {
            if (values.Dimension() == 0)
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot construct a diagonal matrix with an empty vector."
                );
            const std::size_t n = values.Dimension();
            auto result = Identity(n);
            #pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < n; i++)
                result[i][i] = values[i];
            return result;
        }

        template <class OtherType>
        friend class Matrix;
    };
} // namespace Math

#endif