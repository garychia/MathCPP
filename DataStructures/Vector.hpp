#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <initializer_list>
#include <array>
#include <sstream>
#include <cmath>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Tuple.hpp"
#include "Exceptions.hpp"

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
        Vector();

        /*
        Constructor with Initial Size and a Value.
        @param s the initial size of the Vector to be generated.
        @param value the value the Vector will be filled with.
        */
        Vector(std::size_t s, const T &value);

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Vector will store.
        */
        Vector(const std::initializer_list<T> &l);

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the elements this Vector will store.
        */
        template <std::size_t N>
        Vector(const std::array<T, N> &arr);

        /*
        Copy Constructor
        @param other a Container to be copied.
        */
        Vector(const Container<T> &other);

        /*
        Copy Constructor
        @param other a Container to be copied.
        */
        template <class OtherType>
        Vector(const Container<OtherType> &other);

        /*
        Move Constructor
        @param other a Container to be moved.
        */
        Vector(Container<T> &&other);

        /*
        Move Constructor
        @param other a Container to be moved.
        */
        template <class OtherType>
        Vector(Container<OtherType> &&other);

        /*
        Copy Assignment
        @param other a Vector.
        @return a reference to this Vector.
        */
        virtual Vector<T> &operator=(const Container<T> &other) override;

        /*
        Copy Assignment
        @param other a Vector that contains values of a different type.
        @return a reference to this Vector.
        */
        template <class OtherType>
        Vector<T> &operator=(const Container<OtherType> &other);

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        virtual T &operator[](const std::size_t &index);

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        virtual const T &operator[](const std::size_t &index) const override;

        /*
        Returns the dimention of this Vector.
        @return the dimention of this Vector.
        */
        std::size_t Dimension() const;

        /*
        Returns the Euclidean norm of this Vector.
        @return the Euclidean norm of this Vector.
        */
        template <class ReturnType>
        ReturnType Length() const;

        /*
        Returns the Lp Norm of this Vector.
        @return the Lp norm of this Vector.
        */
        template <class ReturnType>
        ReturnType LpNorm(ReturnType p) const;

        /*
        Performs addition with another Vector.
        @param other a Vector to be added.
        @return a Vector that is the result of the addition.
        @throw EmptyVector when this Vector is empty.
        @throw InvalidArgument when the given Vector is empty.
        @throw InvalidArgument when the dimensions of the two vectors mismatch.
        */
        template <class OtherType>
        auto Add(const Vector<OtherType> &other) const;

        /*
        Performs addition with a scaler.
        @param scaler a scaler.
        @return a Vector that is the result of the addition.
        */
        template <class ScalerType>
        auto Add(const ScalerType &scaler) const;

        /*
        Performs addition with another Vector. Reference: Vector.Add.
        @param other a Vector to be added.
        @return a Vector that is the result of the addition.
        */
        template <class OtherType>
        auto operator+(const Vector<OtherType> &other) const;

        /*
        Performs addition with a scaler. Reference: Vector.Add.
        @param scaler a scaler to be added.
        @return a Vector that is the result of the addition.
        */
        template <class ScalerType>
        auto operator+(const ScalerType &scaler) const;

        /*
        Performs inplace addition with another Vector.
        @param other a Vector to be added.
        @return the reference of this Vector.
        */
        template <class OtherType>
        Vector<T> &operator+=(const Vector<OtherType> &other);

        /*
        Performs subtraction with another Vector.
        @param other a Vector to be subtracted.
        @return a Vector that is the result of the subtraction.
        */
        template <class OtherType>
        auto Minus(const Vector<OtherType> &other) const;

        /*
        Performs subtraction with a scaler.
        @param scaler a scaler to be subtracted.
        @return a Vector that is the result of the subtraction.
        */
        template <class ScalerType>
        auto Minus(const ScalerType &scaler) const;

        /*
        Performs subtraction with another Vector. Reference: Vector.Minus.
        @param other a Vector to be subtracted.
        @return a Vector that is the result of the subtraction.
        */
        template <class OtherType>
        auto operator-(const Vector<OtherType> &other) const;

        /*
        Performs subtraction with a scaler. Reference: Vector.Minus.
        @param scaler a scaler to be subtracted.
        @return a Vector that is the result of the subtraction.
        */
        template <class ScalerType>
        auto operator-(const ScalerType &scaler) const;

        /*
        Performs inplace subtraction with another Vector.
        @param other a Vector to be subtracted.
        @return the reference of this Vector.
        */
        template <class OtherType>
        Vector<T> &operator-=(const Vector<OtherType> &other);

        /*
        Performs vector scaling.
        @param scaler a scaler used to scale this Vector.
        @return a Vector that is the result of the scaling.
        */
        template <class OtherType>
        auto Scale(const OtherType &scaler) const;

        /*
        Performs vector scaling. Reference: Vector.Scale.
        @param scaler a scaler used to scale this Vector.
        @return a Vector that is the result of the scaling.
        */
        template <class OtherType>
        auto operator*(const OtherType &scaler) const;

        /*
        Performs vector element-wise multiplication.
        @param other a Vector.
        @return a Vector that is the result of the multiplication.
        @throw EmptyVector when this vector is empty.
        @throw InvalidArgument when the two vectors have different
        dimensions.
        */
        template <class OtherType>
        auto operator*(const Vector<OtherType> &other) const;

        /*
        Performs inplace vector scaling.
        @param scaler a scaler used to scale this Vector.
        @return the reference of this Vector.
        */
        Vector<T> &operator*=(const T &scaler);

        /*
        Performs inplace element-wise vector multiplication.
        @param other a vector.
        @return the reference of this Vector.
        @throw EmptyVector when this vector is empty.
        @throw InvalidArgument when the dimensions of the vectors
        are different.
        */
        Vector<T> &operator*=(const Vector<T> &other);

        /*
        Divides this Vector by a scaler.
        @param scaler a scaler used to divide this Vector.
        @return a Vector that is the result of the division.
        */
        template <class OtherType>
        auto Divide(const OtherType &scaler) const;

        /*
        Divides this Vector by a scaler. Reference: Vector.Divide
        @param scaler a scaler used to divide this Vector.
        @return a Vector that is the result of the division.
        */
        template <class OtherType>
        auto operator/(const OtherType &scaler) const;

        /*
        Performs inplace division on this Vector3D to be divided by a scaler.
        @param scaler a scaler used to divide this Vector3D.
        @return a reference of this Vector3D.
        */
        template <class OtherType>
        Vector<T> &operator/=(const OtherType &scaler);

        /*
        Performs dot product on this Vector with another Vector.
        @param other a Vector.
        @return a scaler that is the dot product.
        */
        template <class OtherType>
        auto Dot(const Vector<OtherType> &other) const;

        /*
        Generates a new vector with normalized values of this Vector.
        @return a normalized vector.
        @thow DividedByZero when this Vector is a zero vector.
        */
        Vector<T> Normalized() const;

        /*
        Normalizes this Vector.
        @throw DividedByZero thown when this vector is a zero vector.
        */
        void Normalize();

        /*
        Calculate the summation of all the elements of this Vector.
        @return the summatoin of the elements.
        */
        T Sum() const;

        /*
        Maps each element of this vector to a new value.
        @param f a function that maps the value of an element to a new value.
        @return a new Vector with the new values defined by f.
        */
        template <class OtherType>
        Vector<OtherType> Map(const std::function<OtherType(T)> &f) const;

        /*
        Returns the pointer that is an array representing this Vector.
        @return a pointer pointing to the first element of this Vector.
        */
        const T *AsRawPointer() const;

        /*
        Generates a vector filled with zeros.
        @param n the number of zeros.
        @return a Vector that is filled with n zeros.
        */
        static Vector<T> ZeroVector(const std::size_t &n);

        /*
        Generates a new Vector with elements from multiple Vectors combined.
        @param vectors a std::initializer_list of Vectors to be combined.
        @return a Vector with the combined elements.
        */
        static Vector<T> Combine(const std::initializer_list<Vector<T>> &vectors);

        template <class ScalerType>
        friend auto operator+(const ScalerType &scaler, const Vector<T> &v)
        {
            Vector<decltype(scaler + v[0])> result(v);
#pragma omp parallel for
            for (std::size_t i = 0; i < result.Dimension(); i++)
                result[i] += scaler;
            return result;
        }

        template <class ScalerType>
        friend auto operator+(const Vector<T> &v, const ScalerType &scaler)
        {
            return scaler + v;
        }

        template <class ScalerType>
        friend auto operator-(const ScalerType &scaler, const Vector<T> &v)
        {
            Vector<decltype(scaler - v[0])> result(v);
#pragma omp parallel for
            for (std::size_t i = 0; i < result.Dimension(); i++)
                result[i] = scaler - result[i];
            return result;
        }

        template <class ScalerType>
        friend auto operator-(const Vector<T> &v, const ScalerType &scaler)
        {
            return v + (-scaler);
        }

        template <class ScalerType>
        friend auto operator*(const ScalerType &scaler, const Vector<T> &v)
        {
            Vector<decltype(scaler * v[0])> result(v);
#pragma omp parallel for
            for (std::size_t i = 0; i < result.Dimension(); i++)
                result[i] *= scaler;
            return result;
        }

        template <class ScalerType>
        friend auto operator/(const ScalerType &scaler, const Vector<T> &v)
        {
            Vector<decltype(scaler / v[0])> result(v);
#pragma omp parallel for
            for (std::size_t i = 0; i < result.Dimension(); i++)
                result[i] = scaler / result[i];
            return result;
        }

        template <class OtherType>
        friend class Vector;
    };
}

#include "Vector.tpp"

#endif