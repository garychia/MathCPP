#ifndef VECTOR_H
#define VECTOR_H

#include <initializer_list>
#include <array>
#include <sstream>
#include <cmath>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tuple.hpp"
#include "../Exceptions/exceptions.hpp"

namespace DataStructure
{
    /*
    Vector is a Container that is capable of storing multiple elements
    such as numbers.
    */
    template <class T>
    class Vector : public Tuple<T>
    {
    public:
        /*
        Constructor that Generates an Empty Vector.
        */
        Vector() : Tuple<T>() {}

        /*
        Constructor with Initial Size and a Value.
        @param s the initial size of the Vector to be generated.
        @param value the value the Vector will be filled with.
        */
        Vector(std::size_t s, const T &value) : Tuple<T>(s, value) {}

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Vector will store.
        */
        Vector(const std::initializer_list<T> &l) : Tuple<T>(l) {}

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the elements this Vector will store.
        */
        template <std::size_t N>
        Vector(const std::array<T, N> &arr) : Tuple<T>(arr) {}

        /*
        Copy Constructor
        @param other a Vector to be copied.
        */
        Vector(const Vector<T> &other) : Tuple<T>(other) {}

        /*
        Copy Constructor
        @param other a Vector to be copied.
        */
        template <class OtherType>
        Vector(const Vector<OtherType> &other) : Tuple<T>(other) {}

        /*
        Move Constructor
        @param other a Vector to be moved.
        */
        Vector(Vector<T> &&other) : Tuple<T>(other) {}

        /*
        Move Constructor
        @param other a Vector to be moved.
        */
        template <class OtherType>
        Vector(Vector<OtherType> &&other) : Tuple<T>(other) {}

        /*
        Copy Assignment
        @param other a Vector.
        @return a reference to this Vector.
        */
        virtual Vector<T> &operator=(const Vector<T> &other)
        {
            Tuple<T>::operator=(other);
            return *this;
        }

        /*
        Copy Assignment
        @param other a Vector that contains values of a different type.
        @return a reference to this Vector.
        */
        template <class OtherType>
        Vector<T> &operator=(const Vector<OtherType> &other)
        {
            Tuple<T>::operator=(other);
            return *this;
        }

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        virtual T &operator[](const std::size_t &index)
        {
            if (index > this->size - 1)
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Vector: Index must be less than the dimention.");
            return this->data[index];
        }

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        virtual const T &operator[](const std::size_t &index) const override
        {
            if (index > this->size - 1)
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Vector: Index must be less than the dimention.");
            return this->data[index];
        }

        /*
        Returns the dimention of this Vector.
        @return the dimention of this Vector.
        */
        std::size_t Dimension() const { return this->size; }

        /*
        Returns the Euclidean norm of this Vector.
        @return the Euclidean norm of this Vector.
        */
        template <class ReturnType>
        ReturnType Length() const
        {
            return LpNorm((ReturnType)2);
        }

        /*
        Returns the Lp Norm of this Vector.
        @return the Lp norm of this Vector.
        */
        template <class ReturnType>
        ReturnType LpNorm(ReturnType p) const
        {
            ReturnType squaredTotal = 0;
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < this->size; i++)
            {
                ReturnType squaredElement = std::pow(this->data[i], p);
#pragma omp atomic
                squaredTotal += squaredElement;
            }
            return std::pow(squaredTotal, 1 / p);
        }

        /*
        Performs addition with another Vector.
        @param other a Vector to be added.
        @return a Vector that is the result of the addition.
        @throw EmptyVector when this Vector is empty.
        @throw InvalidArgument when the given Vector is empty.
        @throw InvalidArgument when the dimensions of the two vectors mismatch.
        */
        template <class OtherType>
        auto Add(const Vector<OtherType> &other) const
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform addition on an empty vector.");
            else if (other.size == 0)
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform addtion on the given empty vector.");
            else if (Dimension() != other.Dimension())
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform addtion on vectors with different dimensions.");
            Vector<decltype(this->data[0] + other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < Dimension(); i++)
                result[i] += other[i];
            return result;
        }

        /*
        Performs addition with another Vector. Reference: Vector.Add.
        @param other a Vector to be added.
        @return a Vector that is the result of the addition.
        */
        template <class OtherType>
        auto operator+(const Vector<OtherType> &other) const
        {
            try
            {
                return this->Add(other);
            }
            catch (Exceptions::EmptyVector &e)
            {
                throw e;
            }
            catch (Exceptions::InvalidArgument &e)
            {
                throw e;
            }
        }

        /*
        Performs inplace addition with another Vector.
        @param other a Vector to be added.
        @return the reference of this Vector.
        */
        template <class OtherType>
        Vector<T> &operator+=(const Vector<OtherType> &other)
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform addition on an empty vector.");
            else if (other.Size() == 0)
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform addtion on the given empty vector.");
            else if (Dimension() != other.Dimension())
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform addtion on vectors with different dimensions.");
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < Dimension(); i++)
                this->data[i] += other[i];
            return *this;
        }

        /*
        Performs subtraction with another Vector.
        @param other a Vector to be subtracted.
        @return a Vector that is the result of the subtraction.
        */
        template <class OtherType>
        auto Minus(const Vector<OtherType> &other) const
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform subtraction on an empty vector.");
            else if (other.size == 0)
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform subtraction on the given empty vector.");
            else if (Dimension() != other.Dimension())
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform subtraction on vectors with different dimensions.");

            Vector<decltype(this->data[0] - other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < Dimension(); i++)
                result[i] -= other[i];
            return result;
        }

        /*
        Performs subtraction with another Vector. Reference: Vector.Minus.
        @param other a Vector to be subtracted.
        @return a Vector that is the result of the subtraction.
        */
        template <class OtherType>
        auto operator-(const Vector<OtherType> &other) const
        {
            try
            {
                return this->Minus(other);
            }
            catch (Exceptions::EmptyVector &e)
            {
                throw e;
            }
            catch (Exceptions::InvalidArgument &e)
            {
                throw e;
            }
        }

        /*
        Performs inplace subtraction with another Vector.
        @param other a Vector to be subtracted.
        @return the reference of this Vector.
        */
        template <class OtherType>
        Vector<T> &operator-=(const Vector<OtherType> &other)
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform subtraction on an empty vector.");
            else if (other.Size() == 0)
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform subtraction on the given empty vector.");
            else if (Dimension() != other.Dimension())
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform subtraction on vectors with different dimensions.");

#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < Dimension(); i++)
                this->data[i] -= other[i];
            return *this;
        }

        /*
        Performs vector scaling.
        @param scaler a scaler used to scale this Vector.
        @return a Vector that is the result of the scaling.
        */
        template <class OtherType>
        auto Scale(const OtherType &scaler) const
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

        /*
        Performs vector scaling. Reference: Vector.Scale.
        @param scaler a scaler used to scale this Vector.
        @return a Vector that is the result of the scaling.
        */
        template <class OtherType>
        auto operator*(const OtherType &scaler) const
        {
            try
            {
                return this->Scale(scaler);
            }
            catch (Exceptions::EmptyVector &e)
            {
                throw e;
            }
        }

        /*
        Performs vector element-wise multiplication.
        @param other a Vector.
        @return a Vector that is the result of the multiplication.
        @throw EmptyVector when this vector is empty.
        @throw InvalidArgument when the two vectors have different
        dimensions.
        */
        template <class OtherType>
        auto operator*(const Vector<OtherType> &other) const
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform scaling on an empty vector.");
            if (this->size != other.Size())
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform elmenet-wise ");
            Vector<decltype((*this)[0] * other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < this->size; i++)
                result[i] *= other[i];
            return result;
        }

        /*
        Performs inplace vector scaling.
        @param scaler a scaler used to scale this Vector.
        @return the reference of this Vector.
        */
        Vector<T> &operator*=(const T &scaler)
        {
            static_assert(!std::is_base_of<Vector, T>::value, "OtherType must be a scaler.");
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform scaling on an empty vector.");
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < Dimension(); i++)
                this->data[i] *= scaler;
            return *this;
        }

        /*
        Performs inplace element-wise vector multiplication.
        @param other a vector.
        @return the reference of this Vector.
        @throw EmptyVector when this vector is empty.
        @throw InvalidArgument when the dimensions of the vectors
        are different.
        */
        Vector<T> &operator*=(const Vector<T> &other)
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform element-wise multiplication on an empty vector.");
            if (Dimension() != other.Dimension())
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform element-wise multiplication on vectors of "
                    "different dimensions.");
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < Dimension(); i++)
                this->data[i] *= other[i];
            return *this;
        }

        /*
        Divides this Vector by a scaler.
        @param scaler a scaler used to divide this Vector.
        @return a Vector that is the result of the division.
        */
        template <class OtherType>
        auto Divide(const OtherType &scaler) const
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform division on an empty vector.");
            else if (scaler == 0)
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot divide a vector by 0.");
            Vector<decltype(this->data[0] / scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < Dimension(); i++)
                result[i] /= scaler;
            return result;
        }

        /*
        Divides this Vector by a scaler. Reference: Vector.Divide
        @param scaler a scaler used to divide this Vector.
        @return a Vector that is the result of the division.
        */
        template <class OtherType>
        auto operator/(const OtherType &scaler) const
        {
            try
            {
                return this->Divide(scaler);
            }
            catch (Exceptions::EmptyVector &e)
            {
                throw e;
            }
            catch (Exceptions::InvalidArgument &e)
            {
                throw e;
            }
        }

        /*
        Performs inplace division on this Vector3D to be divided by a scaler.
        @param scaler a scaler used to divide this Vector3D.
        @return a reference of this Vector3D.
        */
        template <class OtherType>
        Vector<T> &operator/=(const OtherType &scaler)
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform division on an empty vector.");
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < Dimension(); i++)
                this->data[i] /= scaler;
            return *this;
        }

        /*
        Performs dot product on this Vector with another Vector.
        @param other a Vector.
        @return a scaler that is the dot product.
        */
        template <class OtherType>
        auto Dot(const Vector<OtherType> &other) const
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform subtraction on an empty vector.");
            else if (other.size == 0)
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform subtraction on the given empty vector.");
            else if (Dimension() != other.Dimension())
                throw Exceptions::InvalidArgument(
                    "Vector: Cannot perform subtraction on vectors with different dimensions.");
            decltype(this->data[0] * other[0]) result = 0;
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < Dimension(); i++)
            {
                decltype(result) mult = this->data[i] * other[i];
#pragma omp atomic
                result += mult;
            }
            return result;
        }

        /*
        Generates a new vector with normalized values of this Vector.
        @return a normalized vector.
        @thow DividedByZero when this Vector is a zero vector.
        */
        Vector<T> Normalized() const
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform normalization on an empty vector.");
            const T length = Length();
            if (length == 0)
                throw Exceptions::DividedByZero("Vector: Cannot normalize a zero vector.");
            return *this / length;
        }

        /*
        Normalizes this Vector.
        @throw DividedByZero thown when this vector is a zero vector.
        */
        void Normalize()
        {
            if (this->size == 0)
                throw Exceptions::EmptyVector(
                    "Vector: Cannot perform normalization on an empty vector.");
            const T length = Length();
            if (length == 0)
                throw Exceptions::DividedByZero("Vector: Cannot normalize a zero vector.");
            *this /= length;
        }

        /*
        Calculate the summation of all the elements of this Vector.
        @return the summatoin of the elements.
        */
        T Sum() const
        {
            T total = 0;
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < this->size; i++)
                total += (*this)[i];
            return total;
        }

        /*
        Maps each element of this vector to a new value.
        @param f a function that maps the value of an element to a new value.
        @return a new Vector with the new values defined by f.
        */
        Vector<T> Map(const std::function<T(T)> &f) const
        {
            Vector<T> result(*this);
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < this->size; i++)
                result[i] = f(result[i]);
            return result;
        }

        /*
        Generates a vector filled with zeros.
        @param n the number of zeros.
        @return a Vector that is filled with n zeros.
        */
        static Vector<T> ZeroVector(const std::size_t &n)
        {
            Vector<T> v;
            v.size = n;
            v.data = new T[n];
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < n; i++)
                v.data[i] = 0;
            return v;
        }

        /*
        Generates a new Vector with elements from multiple Vectors combined.
        @param vectors a std::initializer_list of Vectors to be combined.
        @return a Vector with the combined elements.
        */
        static Vector<T> Combine(const std::initializer_list<Vector<T>> &vectors)
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

        template <class OtherType>
        friend class Vector;
    };
}

#endif