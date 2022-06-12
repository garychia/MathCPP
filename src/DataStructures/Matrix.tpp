#include "Exceptions.hpp"
#include "Math.hpp"

namespace DataStructures
{

    template <class T>
    Matrix<T>::Matrix() : Container<Vector<T>>(), nRows(0), nColumns(0)
    {
    }

    template <class T>
    Matrix<T>::Matrix(std::size_t numRows, std::size_t numColumns, T initialValue)
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

    template <class T>
    Matrix<T>::Matrix(std::initializer_list<Vector<T>> l) : Container<Vector<T>>(l)
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

    template <class T>
    Matrix<T>::Matrix(std::initializer_list<T> l, bool column)
    {
        if (l.size() > 0)
        {
            if (!column)
            {
                this->size = 1;
                this->data = new Vector<T>[1];
                this->data[0] = Vector<T>(l);
                nRows = 1;
                nColumns = l.size();
            }
            else
            {
                this->size = l.size();
                this->data = new Vector<T>[l.size()];
                nRows = l.size();
                nColumns = 1;
                std::size_t index = 0;
                for (auto itr = l.begin(); itr != l.end(); itr++)
                    this->data[index++] = Vector<T>({*itr});
            }
        }
        else
        {
            this->size = 0;
            this->data = nullptr;
            nRows = 0;
            nColumns = 0;
        }
    }

    template <class T>
    template <std::size_t N>
    Matrix<T>::Matrix(const std::array<Vector<T>, N> &arr)
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

    template <class T>
    Matrix<T>::Matrix(const List<Vector<T>> &l)
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

    template <class T>
    Matrix<T>::Matrix(const Matrix<T> &other) : Container<Vector<T>>(other)
    {
        nRows = other.nRows;
        nColumns = (nRows > 0) ? this->data[0].Size() : 0;
    }

    template <class T>
    template <class OtherType>
    Matrix<T>::Matrix(const Matrix<OtherType> &other) : Container<Vector<T>>(other)
    {
        nRows = other.Shape()[0];
        nColumns = (nRows > 0) ? this->data[0].Size() : 0;
    }

    template <class T>
    Matrix<T>::Matrix(Matrix<T> &&other) : Container<Vector<T>>(other)
    {
        nRows = std::move(other.nRows);
        nColumns = (nRows > 0) ? (*this)[0].Size() : 0;
        other.nRows = 0;
        other.nColumns = 0;
    }

    template <class T>
    template <class OtherType>
    Matrix<T>::Matrix(Matrix<OtherType> &&other) : Container<Vector<T>>(other)
    {
        nRows = std::move(other.Shape()[0]);
        nColumns = (nRows > 0) ? (*this)[0].Size() : 0;
        other.nRows = 0;
        other.nColumns = 0;
    }

    template <class T>
    Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
    {
        if (this != &other)
        {
            Container<Vector<T>>::operator=(other);
            nRows = other.nRows;
            nColumns = other.nColumns;
        }
        return *this;
    }

    template <class T>
    Vector<T> &Matrix<T>::operator[](const std::size_t &index)
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

    template <class T>
    const Vector<T> &Matrix<T>::operator[](const std::size_t &index) const
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

    template <class T>
    std::size_t Matrix<T>::Size() const { return this->size; }

    template <class T>
    Tuple<std::size_t> Matrix<T>::Shape() const { return Tuple<std::size_t>({nRows, nColumns}); }

    template <class T>
    template <class OtherType>
    auto Matrix<T>::Add(const Matrix<OtherType> &other) const
    {
        if (other.IsEmpty())
            throw Exceptions::InvalidArgument(
                "Matrix: Cannot perform addition with an empty matrix.");
        else if (this->IsEmpty())
            throw Exceptions::EmptyMatrix(
                "Matrix: Cannot perform addition with an empty matrix.");
        const auto otherShape = other.Shape();
        const auto thisShape = Shape();
        if (!(thisShape[0] % otherShape[0] == 0 && thisShape[1] % otherShape[1] == 0))
        {
            std::stringstream errorMessageStream;
            errorMessageStream
                << "Expected the numbers of rows and columns of the second matrix"
                   " to be factors of these of the first matrix.";
            throw Exceptions::MatrixShapeMismatch(
                thisShape,
                otherShape,
                errorMessageStream.str());
        }

        Matrix<decltype((*this)[0][0] + other[0][0])> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nRows; i++)
            result[i] += other[i % other.nRows];
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Matrix<T>::operator+(const Matrix<OtherType> &other) const
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

    template <class T>
    template <class OtherType>
    Matrix<T> &Matrix<T>::operator+=(const Matrix<OtherType> &other)
    {
        if (other.IsEmpty())
            throw Exceptions::InvalidArgument(
                "Matrix: Cannot perform addition with an empty matrix.");
        else if (this->IsEmpty())
            throw Exceptions::EmptyMatrix(
                "Matrix: Cannot perform addition with an empty matrix.");
        const auto otherShape = other.Shape();
        const auto thisShape = Shape();
        if (!(thisShape[0] % otherShape[0] == 0 && thisShape[1] % otherShape[1] == 0))
        {
            std::stringstream errorMessageStream;
            errorMessageStream
                << "Expected the numbers of rows and columns of the second matrix"
                   " to be factors of these of the first matrix."
                << std::endl;
            throw Exceptions::MatrixShapeMismatch(
                thisShape,
                otherShape,
                errorMessageStream.str());
        }

#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nRows; i++)
            (*this)[i] += other[i % other.nRows];
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Matrix<T>::Subtract(const Matrix<OtherType> &other) const
    {
        if (other.IsEmpty())
            throw Exceptions::InvalidArgument(
                "Matrix: Cannot perform subtraction with an empty matrix.");
        else if (this->IsEmpty())
            throw Exceptions::EmptyMatrix(
                "Matrix: Cannot perform subtraction with an empty matrix.");
        const auto otherShape = other.Shape();
        const auto thisShape = Shape();
        if (!(thisShape[0] % otherShape[0] == 0 && thisShape[1] % otherShape[1] == 0))
        {
            std::stringstream errorMessageStream;
            errorMessageStream
                << "Expected the numbers of rows and columns of the second matrix"
                   " to be factors of these of the first matrix."
                << std::endl;
            throw Exceptions::MatrixShapeMismatch(
                thisShape,
                otherShape,
                errorMessageStream.str());
        }

        Matrix<decltype((*this)[0][0] + other[0][0])> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nRows; i++)
            result[i] -= other[i % other.nRows];
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Matrix<T>::operator-(const Matrix<OtherType> &other) const
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

    template <class T>
    template <class OtherType>
    Matrix<T> &Matrix<T>::operator-=(const Matrix<OtherType> &other)
    {
        if (other.IsEmpty())
            throw Exceptions::InvalidArgument(
                "Matrix: Cannot perform subtraction with an empty matrix.");
        else if (this->IsEmpty())
            throw Exceptions::EmptyMatrix(
                "Matrix: Cannot perform subtraction with an empty matrix.");
        const auto otherShape = other.Shape();
        const auto thisShape = Shape();
        if (!(thisShape[0] % otherShape[0] == 0 && thisShape[1] % otherShape[1] == 0))
        {
            std::stringstream errorMessageStream;
            errorMessageStream
                << "Expected the numbers of rows and columns of the second matrix"
                   " to be factors of these of the first matrix."
                << std::endl;
            throw Exceptions::MatrixShapeMismatch(
                thisShape,
                otherShape,
                errorMessageStream.str());
        }

#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nRows; i++)
            (*this)[i] -= other[i % other.nRows];
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Matrix<T>::Multiply(const Matrix<OtherType> &other) const
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
                << ".";
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

    template <class T>
    template <class OtherType>
    auto Matrix<T>::operator*(const Matrix<OtherType> &other) const
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

    template <class T>
    auto Matrix<T>::Scale(const T &scaler) const
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

    template <class T>
    template <class OtherType>
    auto Matrix<T>::Scale(const Matrix<OtherType> &other) const
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
                << "Expected two matrices have the same shape when performing "
                   "element-wise scaling.";
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

    template <class T>
    auto Matrix<T>::operator*(const T &scaler) const
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

    template <class T>
    Matrix<T> &Matrix<T>::operator*=(const T &scaler)
    {
        if (this->IsEmpty())
            throw Exceptions::EmptyMatrix(
                "Matrix: Cannot perform scaling on an empty matrix.");

#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nRows; i++)
            (*this)[i] *= scaler;
        return *this;
    }

    template <class T>
    Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &other)
    {
        if (this->IsEmpty())
            throw Exceptions::EmptyMatrix(
                "Matrix: Cannot perform scaling on an empty matrix.");

#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nRows; i++)
            (*this)[i] *= other[i];
        return *this;
    }

    template <class T>
    template <class ScalerType>
    auto Matrix<T>::Divide(const ScalerType &scaler) const
    {
        if (scaler == 0)
            throw Exceptions::DividedByZero(
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

    template <class T>
    template <class OtherType>
    auto Matrix<T>::Divide(const Matrix<OtherType> &matrix) const
    {
        if (this->IsEmpty() || matrix.IsEmpty())
            throw Exceptions::EmptyMatrix(
                "Matrix: Cannot perform element-wise division on an empty matrix.");
        if (nRows % matrix.nRows != 0 || nColumns % matrix.nColumns != 0)
            throw Exceptions::InvalidArgument(
                "Matrix: The shape of the denominator matrix must be a factor of that of the numerator matrix.");
        Matrix<decltype((*this)[0][0] / matrix[0][0])> result(*this);
        std::size_t j;
#pragma omp parallel for schedule(dynamic) private(j)
        for (std::size_t i = 0; i < nRows; i++)
        {
            j = i % matrix.nRows;
            try
            {
                result[i] /= matrix[j];
            }
            catch (const Exceptions::DividedByZero &e)
            {
                throw Exceptions::DividedByZero(
                    "Matrix: Zero denominator found when performing division.");
            }
        }
        return result;
    }

    template <class T>
    template <class ScalerType>
    auto Matrix<T>::operator/(const ScalerType &scaler) const
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

    template <class T>
    Matrix<T> &Matrix<T>::operator/=(const T &scaler)
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

    template <class T>
    std::string Matrix<T>::ToString() const
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
    template <class T>
    std::ostream &operator<<(std::ostream &os, const Matrix<T> &m)
    {
        os << m.ToString();
        return os;
    }

    template <class T>
    void Matrix<T>::Transpose()
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

    template <class T>
    Matrix<T> Matrix<T>::Transposed() const
    {
        Matrix<T> transpose(nColumns, nRows);
        std::size_t i, j;
#pragma omp parallel for private(j) schedule(dynamic) collapse(2)
        for (i = 0; i < nColumns; i++)
            for (j = 0; j < nRows; j++)
                transpose[i][j] = this->data[j][i];
        return transpose;
    }

    template <class T>
    Matrix<T> Matrix<T>::Flattened(bool rowMajor, bool keepInRow) const
    {
        auto flattened = keepInRow ? Matrix<T>(1, nRows * nColumns) : Matrix<T>(nRows * nColumns, 1);
        std::size_t i, j;
        if (rowMajor)
        {
#pragma omp parallel for private(j) schedule(dynamic) collapse(2)
            for (i = 0; i < nRows; i++)
                for (j = 0; j < nColumns; j++)
                {
                    if (keepInRow)
                        flattened[0][i * nColumns + j] = (*this)[i][j];
                    else
                        flattened[i * nColumns + j][0] = (*this)[i][j];
                }
            return flattened;
        }
#pragma omp parallel for private(j) schedule(dynamic) collapse(2)
        for (i = 0; i < nColumns; i++)
            for (j = 0; j < nRows; j++)
            {
                if (keepInRow)
                    flattened[0][i * nRows + j] = (*this)[j][i];
                else
                    flattened[i * nRows + j][0] = (*this)[j][i];
            }
        return flattened;
    }

    template <class T>
    T Matrix<T>::SumAll() const
    {
        T total = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ \
                                                     : total)
        for (std::size_t i = 0; i < nRows; i++)
            total += (*this)[i].Sum();
        return total;
    }

    template <class T>
    Matrix<T> Matrix<T>::Sum(bool sumRows) const
    {
        if (this->IsEmpty())
            throw Exceptions::EmptyMatrix(
                "Matrix: Cannot perform summmation on an empty matrix.");
        if (sumRows)
        {
            Vector<T> summation = (*this)[0];
            for (std::size_t i = 1; i < nRows; i++)
                summation += (*this)[i];
            return Matrix<T>({summation});
        }

        Matrix<T> summation(nRows, 1);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nRows; i++)
            summation[i][0] = (*this)[i].Sum();
        return summation;
    }

    template <class T>
    template <class ReturnType>
    ReturnType Matrix<T>::FrobeniusNorm() const
    {
        return Math::Power<T, double>(Map([](const T &e) { return e * e; }).SumAll(), 0.5);
    }

    template <class T>
    template <class MapFunction>
    auto Matrix<T>::Map(MapFunction &&f) const
    {
        Matrix<decltype(f((*this)[0][0]))> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nRows; i++)
            result[i] = result[i].Map(f);
        return result;
    }

    template <class T>
    Matrix<T> Matrix<T>::Identity(std::size_t n)
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

    template <class T>
    Matrix<T> Matrix<T>::Diagonal(const Vector<T> &values)
    {
        if (values.Dimension() == 0)
            throw Exceptions::InvalidArgument(
                "Matrix: Cannot construct a diagonal matrix with an empty vector.");
        const std::size_t n = values.Dimension();
        auto result = Identity(n);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < n; i++)
            result[i][i] = values[i];
        return result;
    }

    template <class T>
    Matrix<T> Matrix<T>::Translation(const Vector<T> &deltas)
    {
        const std::size_t n = deltas.Size() + 1;
        auto translationM = Identity(n);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < n - 1; i++)
            translationM[i][n - 1] = deltas[i];
        return translationM;
    }

    template <class T>
    Matrix<T> Matrix<T>::Scaling(const Vector<T> &factors)
    {
        const std::size_t n = factors.Size() + 1;
        auto scalingM = Identity(n);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < n - 1; i++)
            scalingM[i][i] *= factors[i];
        return scalingM;
    }

    template <class T>
    Matrix<T> Matrix<T>::Rotation2D(const T &radians)
    {
        const T sinValue = Math::Sine(radians);
        const T cosValue = Math::Cosine(radians);
        return Matrix<T>(
            {{cosValue, -sinValue, 0},
             {sinValue, cosValue, 0},
             {0, 0, 1}});
    }

    template <class T>
    Matrix<T> Matrix<T>::Rotation3D(const Vector<T> &axis, const T &radians)
    {
        if (axis.Dimension() != 3)
        {
            std::stringstream ss;
            ss << "Matrix: Rotation3D requires an axis defined in 3D space "
                  "but got an axis of dimension "
               << axis.Dimension()
               << ".";
            throw Exceptions::InvalidArgument(ss.str());
        }
        const auto normalizedAxis = axis.Normalized();
        const T &x = normalizedAxis[0];
        const T &y = normalizedAxis[1];
        const T &z = normalizedAxis[2];
        const T sinValue = Math::Sine(radians);
        const T cosValue = Math::Cosine(radians);
        const T oneMinusCosValue = 1 - cosValue;
        return Matrix<T>(
            {{cosValue + x * x * oneMinusCosValue, x * y * oneMinusCosValue - z * sinValue, x * z * oneMinusCosValue + y * sinValue, 0},
             {y * x * oneMinusCosValue + z * sinValue, cosValue + y * y * oneMinusCosValue, y * z * oneMinusCosValue - x * sinValue, 0},
             {z * x * oneMinusCosValue - y * sinValue, z * y * oneMinusCosValue + x * sinValue, cosValue + z * z * oneMinusCosValue, 0},
             {0, 0, 0, 1}});
    }

    template <class T>
    Matrix<T> Matrix<T>::Perspective(T fov, T aspect, T near, T far)
    {
        if (fov <= 0)
        {
            std::stringstream ss;
            ss << "Matrix::Perspective: Field of View must be positive but got " << fov;
            throw Exceptions::InvalidArgument(ss.str());
        }
        if (aspect == 0)
        {
            std::stringstream ss;
            ss << "Matrix::Perspective: Aspect Ratio must be non-zero but got " << aspect;
            throw Exceptions::InvalidArgument(ss.str());
        }
        const T scale = 1 / Math::Tangent(fov * 0.5);
        const T farNearDiff = far - near;
        return Matrix<T>({{scale * aspect, 0, 0, 0},
                          {0, scale, 0, 0},
                          {0, 0, -(far + near) / farNearDiff, -2 * near * far / farNearDiff},
                          {0, 0, -1, 0}});
    }

    template <class T>
    Matrix<T> Matrix<T>::Orthographic(T left, T right, T bottom, T top, T near, T far)
    {
        if (left == right)
            throw Exceptions::InvalidArgument("Matrix::Orthographic: 'left' and 'right' should not be the same value.");
        if (bottom == top)
            throw Exceptions::InvalidArgument("Matrix::Orthographic: 'top' and 'bottom' should not be the same value.");
        if (near == far)
            throw Exceptions::InvalidArgument("Matrix::Orthographic: 'near' and 'far' should not be the same value.");
        const T rightLeftDiff = right - left;
        const T topBottomDiff = top - bottom;
        const T farNearDist = far - near;
        return Matrix<T>({{2 / rightLeftDiff, 0, 0, -(right + left) / rightLeftDiff},
                          {0, 2 / topBottomDiff, 0, -(top + bottom) / topBottomDiff},
                          {0, 0, -2 / farNearDist, -(far + near) / farNearDist},
                          {0, 0, 0, 1}});
    }
} // namespace DataStructure