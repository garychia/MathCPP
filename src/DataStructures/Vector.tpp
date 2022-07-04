#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDA_ENABLED__
#include "CudaHelpers.hpp"
#endif
#include "Exceptions.hpp"
#include "Math.hpp"

namespace DataStructures
{
    template <class T>
    Vector<T>::Vector() : Tuple<T>() {}

    template <class T>
    Vector<T>::Vector(std::size_t s, const T &value) : Tuple<T>(s, value) {}

    template <class T>
    Vector<T>::Vector(const std::initializer_list<T> &l) : Tuple<T>(l) {}

    template <class T>
    template <std::size_t N>
    Vector<T>::Vector(const std::array<T, N> &arr) : Tuple<T>(arr) {}

    template <class T>
    Vector<T>::Vector(const std::vector<T> &v) : Tuple<T>(v) {}

    template <class T>
    Vector<T>::Vector(const Container<T> &other) : Tuple<T>(other) {}

    template <class T>
    template <class OtherType>
    Vector<T>::Vector(const Container<OtherType> &other) : Tuple<T>(other) {}

    template <class T>
    Vector<T>::Vector(Container<T> &&other) : Tuple<T>(other) {}

    template <class T>
    Vector<T> &Vector<T>::operator=(const Container<T> &other)
    {
        Tuple<T>::operator=(other);
        return *this;
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator=(const Container<OtherType> &other)
    {
        Tuple<T>::operator=(other);
        return *this;
    }

    template <class T>
    T &Vector<T>::operator[](const std::size_t &index)
    {
        if (index > this->size - 1)
            throw Exceptions::IndexOutOfBound(
                index,
                "Vector: Index must be less than the dimension.");
        return this->data[index];
    }

    template <class T>
    const T &Vector<T>::operator[](const std::size_t &index) const
    {
        if (index > this->size - 1)
            throw Exceptions::IndexOutOfBound(
                index,
                "Vector: Index must be less than the dimension.");
        return this->data[index];
    }

    template <class T>
    std::size_t Vector<T>::Dimension() const { return this->size; }

    template <class T>
    template <class ReturnType>
    ReturnType Vector<T>::Length() const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Length of an empty vector is undefined.");
        return LpNorm<ReturnType>(2);
    }

    template <class T>
    template <class ReturnType>
    ReturnType Vector<T>::EuclideanNorm() const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Euclidean norm of an empty vector is undefined.");
        return Length<ReturnType>();
    }

    template <class T>
    template <class ReturnType>
    ReturnType Vector<T>::LpNorm(int p) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Lp norm of an empty vector is undefined.");
        ReturnType squaredTotal = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : squaredTotal)
        for (std::size_t i = 0; i < Dimension(); i++)
            squaredTotal += Math::Power<ReturnType, int>((*this)[i], p);
        return Math::Power<ReturnType, double>(squaredTotal, (double)1 / p);
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Add(const Vector<OtherType> &other) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform addition on an empty vector.");
        else if (other.size == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform addtion on the given empty vector.");
        else if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.");
        Vector<decltype(this->data[0] + other[0])> result(*this);
#ifdef __CUDA_ENABLED__
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i += other.Dimension())
            CudaHelpers::AddWithTwoArrays(&result[i], &(*this)[i], other.data, other.Dimension());
#else
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] += other[i % other.Dimension()];
#endif
        return result;
    }

    template <class T>
    template <class ScalerType>
    auto Vector<T>::Add(const ScalerType &scaler) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform addition on an empty vector.");
        Vector<decltype(this->data[0] + scaler)> result(*this);
#ifdef __CUDA_ENABLED__
        CudaHelpers::AddWithArrayScaler(&result[0], &(*this)[0], scaler, result.Dimension());
#else
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] += scaler;
#endif
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator+(const Vector<OtherType> &other) const
    {
        return this->Add(other);
    }

    template <class T>
    template <class ScalerType>
    auto Vector<T>::operator+(const ScalerType &scaler) const
    {
        return this->Add(scaler);
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator+=(const Vector<OtherType> &other)
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform addition on an empty vector.");
        else if (other.Size() == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform addtion on the given empty vector.");
        else if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            this->data[i] += other[i % other.Dimension()];
        return *this;
    }

    template <class T>
    template <class ScalerType>
    Vector<T> &Vector<T>::operator+=(const ScalerType &scaler)
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform addition on an empty vector.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            this->data[i] += scaler;
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Minus(const Vector<OtherType> &other) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform subtraction on an empty vector.");
        else if (other.size == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform subtraction on the given empty vector.");
        else if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.");
        Vector<decltype(this->data[0] - other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] -= other[i % other.Dimension()];
        return result;
    }

    template <class T>
    template <class ScalerType>
    auto Vector<T>::Minus(const ScalerType &scaler) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform subtraction on an empty vector.");
        Vector<decltype(this->data[0] - scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] -= scaler;
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator-(const Vector<OtherType> &other) const
    {
        return this->Minus(other);
    }

    template <class T>
    template <class ScalerType>
    auto Vector<T>::operator-(const ScalerType &scaler) const
    {
        return this->Minus(scaler);
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator-=(const Vector<OtherType> &other)
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform subtraction on an empty vector.");
        else if (other.Size() == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform subtraction on the given empty vector.");
        else if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            this->data[i] -= other[i % other.Dimension()];
        return *this;
    }

    template <class T>
    template <class ScalerType>
    Vector<T> &Vector<T>::operator-=(const ScalerType &scaler)
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform subtraction on an empty vector.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            this->data[i] -= scaler;
        return *this;
    }

    template <class T>
    template <class ScalerType>
    auto Vector<T>::Scale(const ScalerType &scaler) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform scaling on an empty vector.");
        Vector<decltype(this->data[0] * scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] *= scaler;
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Scale(const Vector<OtherType> &other) const
    {
        if (Dimension() == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform scaling on an empty vector.");
        else if (other.Dimension() == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform scaling with an empty vector as the second operand.");
        else if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.");
        Vector<decltype((*this)[0] * other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] *= other[i % other.Dimension()];
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator*(const OtherType &scaler) const
    {
        return this->Scale(scaler);
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator*(const Vector<OtherType> &other) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform element-wise multiplication on an empty vector.");
        if (other.Dimension() == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform element-wise multiplication with an empty vector as the second operand.");
        if (this->Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expect the dimension of the second operand is a factor of that "
                "of the first operand when performing element-wise multiplication.");
        Vector<decltype((*this)[0] * other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < this->size; i++)
            result[i] *= other[i % other.Dimension()];
        return result;
    }

    template <class T>
    template <class ScalerType>
    Vector<T> &Vector<T>::operator*=(const ScalerType &scaler)
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform scaling on an empty vector.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            (*this)[i] *= scaler;
        return *this;
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator*=(const Vector<OtherType> &other)
    {
        if (Dimension() == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform element-wise multiplication on an empty vector.");
        if (!other.Dimension())
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform element-wise multiplication with an empty vector as the second operand.");
        if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expect the dimension of the second operand is a factor of that "
                "of the first operand when performing element-wise multiplication.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            (*this)[i] *= other[i % other.Dimension()];
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Divide(const OtherType &scaler) const
    {
        if (Dimension() == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform element-wise division on an empty vector.");
        else if (scaler == 0)
            throw Exceptions::DividedByZero(
                "Vector: Cannot perform element-wise division as the second operand is 0.");
        Vector<decltype((*this)[0] / scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] /= scaler;
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Divide(const Vector<OtherType> &vector) const
    {
        if (Dimension() == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform element-wise division on an empty vector.");
        else if (vector.Dimension() == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform element-wise division as the second operand is empty.");
        else if (Dimension() % vector.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform element-wise division. Expected the dimension of the "
                "second operand to be a factor of that of the first operand.");
        Vector<decltype((*this)[0] / vector[0])> result(*this);
        std::size_t j;
#pragma omp parallel for schedule(dynamic) private(j)
        for (std::size_t i = 0; i < Dimension(); i++)
        {
            j = i % vector.Dimension();
            if (vector[j] == 0)
                throw Exceptions::DividedByZero(
                    "Vector: Expect none of the element of the second operand to be 0 when performing"
                    "element-wise division.");
            result[i] /= vector[j];
        }
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator/(const OtherType &scaler) const
    {
        return this->Divide(scaler);
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator/(const Vector<OtherType> &vector) const
    {
        return this->Divide(vector);
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator/=(const OtherType &scaler)
    {
        if (Dimension() == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform element-wise division on an empty vector.");
        else if (scaler == 0)
            throw Exceptions::DividedByZero("Vector: Cannot perform element-wise division as the second operand is 0.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            (*this)[i] /= scaler;
        return *this;
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator/=(const Vector<OtherType> &vector)
    {
        if (Dimension() == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform element-wise division on an empty vector.");
        else if (vector.Dimension() == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform element-wise division when the second operand is empty.");
        else if (Dimension() % vector.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform element-wise division. Expected the dimension of the "
                "second operand to be a factor of that of the first operand.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
        {
            const auto element = vector[i % vector.Dimension()];
            if (element == 0)
                throw Exceptions::DividedByZero(
                    "Vector: Expect none of the element of the second operand to be 0 when performing"
                    "element-wise division.");
            (*this)[i] /= element;
        }
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Dot(const Vector<OtherType> &other) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform dot product on an empty vector.");
        else if (other.size == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform dot product when the second operand is empty.");
        else if (Dimension() != other.Dimension())
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform dot product on vectors with different dimensions.");
        decltype((*this)[0] * other[0]) result = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ \
                                                     : result)
        for (std::size_t i = 0; i < Dimension(); i++)
            result += (*this)[i] * other[i];
        return result;
    }

    template <class T>
    template <class ReturnType>
    Vector<ReturnType> Vector<T>::Normalized() const
    {
        if (Dimension() == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform normalization on an empty vector.");
        const ReturnType length = Length<ReturnType>();
        if (length == 0)
            throw Exceptions::DividedByZero("Vector: Cannot normalize a zero vector.");
        return *this / length;
    }

    template <class T>
    void Vector<T>::Normalize()
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform normalization on an empty vector.");
        const T length = Length<T>();
        if (length == 0)
            throw Exceptions::DividedByZero("Vector: Cannot normalize a zero vector.");
        *this /= length;
    }

    template <class T>
    T Vector<T>::Sum() const
    {
        T total = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : total)
        for (std::size_t i = 0; i < this->size; i++)
            total += (*this)[i];
        return total;
    }

    template <class T>
    template <class MapFunction>
    auto Vector<T>::Map(MapFunction &&f) const
    {
        Vector<decltype(f((*this)[0]))> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < this->size; i++)
            result[i] = f(result[i]);
        return result;
    }

    template <class T>
    const T *Vector<T>::AsRawPointer() const { return this->data; }

    template <class T>
    Vector<T> Vector<T>::ZeroVector(const std::size_t &n)
    {
        return Vector<T>(n, 0);
    }

    template <class T>
    Vector<T> Vector<T>::Combine(const std::initializer_list<Vector<T>> &vectors)
    {
        std::size_t elementTotal = 0;
        for (auto itr = vectors.begin(); itr != vectors.end(); itr++)
        {
            elementTotal += itr->Size();
        }
        Vector<T> combined(elementTotal, 0);
        std::size_t currentIndex = 0;
        for (auto vector : vectors)
            for (std::size_t j = 0; j < vector.Size(); j++)
                combined[currentIndex++] = vector[j];
        return combined;
    }

    template <class T, class ScalerType>
    auto operator+(const ScalerType &scaler, const Vector<T> &v)
    {
        Vector<decltype(scaler + v[0])> result(v);
#pragma omp parallel for
        for (std::size_t i = 0; i < result.Dimension(); i++)
            result[i] += scaler;
        return result;
    }

    template <class T, class ScalerType>
    auto operator-(const ScalerType &scaler, const Vector<T> &v)
    {
        Vector<decltype(scaler - v[0])> result(v);
#pragma omp parallel for
        for (std::size_t i = 0; i < result.Dimension(); i++)
            result[i] = scaler - result[i];
        return result;
    }

    template <class T, class ScalerType>
    auto operator*(const ScalerType &scaler, const Vector<T> &v)
    {
        Vector<decltype(scaler * v[0])> result(v);
#pragma omp parallel for
        for (std::size_t i = 0; i < result.Dimension(); i++)
            result[i] *= scaler;
        return result;
    }

    template <class T, class ScalerType>
    auto operator/(const ScalerType &scaler, const Vector<T> &v)
    {
        Vector<decltype(scaler / v[0])> result(v);
#pragma omp parallel for
        for (std::size_t i = 0; i < result.Dimension(); i++)
            result[i] = scaler / result[i];
        return result;
    }
}