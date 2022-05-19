#ifndef VECTOR3D_HPP
#define VECTOR3D_HPP

#include <cmath>
#include <sstream>
#include <ostream>

#include "container.hpp"
#include "../Exceptions/exceptions.hpp"

#define NUM_COMPONENTS 3
#define X_INDEX 0
#define Y_INDEX 1
#define Z_INDEX 2

namespace DataStructure
{
    /*
    Vector3D is a class that represents a vector in a 3D space.
    */
    template <class T>
    class Vector3D : public Container<T>
    {
    public:
        /*
        Retrieves the X component of this vector.
        @return the X component.
        */
        T &X();
        /*
        Retrieves the Y component of this vector.
        @return the Y component.
        */
        T &Y();
        /*
        Retrieves the Z component of this vector.
        @return the Z component.
        */
        T &Z();

        /*
        Constructor with Component X, Y and Z
        @param x the X component of this vector
        @param y the y component of this vector
        @param z the z component of this vector
        */
        Vector3D(T x = 0, T y = 0, T z = 0);

        /*
        Copy Constructor
        @param other a Vector3D to be copied.
        */
        Vector3D(const Vector3D<T> &other);

        /*
        Copy Constructor
        @param other a Vector3D to be copied.
        */
        template<class OtherType>
        Vector3D(const Vector3D<OtherType> &other);

        /*
        Move Constructor
        @param other a Vector3D to be moved.
        */
        Vector3D(Vector3D<T> &&other);

        /*
        Move Constructor
        @param other a Vector3D to be moved.
        */
        template<class OtherType>
        Vector3D(Vector3D<T> &&other);

        /*
        Copy Assignment
        @param other a Vector3D to be copied.
        @return a reference to this Vector3D.
        */
        virtual Vector3D<T> &operator=(const Vector3D<T> &other);

        /*
        Returns a string that represents this Vector3D.
        @return a string that contains information about this Vector3D.
        */
        virtual std::string ToString() const override;

        /*
        Returns a component of this Vector3D.
        @param index the index of the X, Y or Z component.
        @return X, Y or Z component when index is 0, 1 or 2, respectively.
        @throw IndexOutOfBound an exception thrown when index is not valid.
        */
        virtual T &operator[](const std::size_t &index);

        /*
        Returns a component of this Vector3D.
        @param index the index of the X, Y or Z component.
        @return X, Y or Z component when index is 0, 1 or 2, respectively.
        @throw IndexOutOfBound an exception thrown when index is not valid.
        */
        virtual const T &operator[](const std::size_t &index) const override;

        /*
        Returns the number of elements this Vector3D has.
        @return always 3 (X, Y and Z components).
        */
        virtual std::size_t Size() const override;

        /*
        Calculates the length of this Vector3D.
        @return the length of this Vector3D.
        */
        template<class ReturnType>
        ReturnType Length() const;

        /*
        Performs addition with another Vector3D.
        @param other a Vector3D to be added.
        @return a Vector3D that is the result of the addition.
        */
        template <class OtherType>
        auto Add(const Vector3D<OtherType> &other) const;

        /*
        Performs addition with another Vector3D. Reference: Vector3D.Add.
        @param other a Vector3D to be added.
        @return a Vector3D that is the result of the addition.
        */
        template <class OtherType>
        auto operator+(const Vector3D<OtherType> &other) const;

        /*
        Performs inplace addition with another Vector3D.
        @param other a Vector3D to be added.
        @return the reference of this Vector3D.
        */
        template <class OtherType>
        Vector3D<T> &operator+=(const Vector3D<OtherType> &other);

        /*
        Performs subtraction with another Vector3D.
        @param other a Vector3D to be subtracted.
        @return a Vector3D that is the result of the subtraction.
        */
        template <class OtherType>
        auto Minus(const Vector3D<OtherType> &other) const;

        /*
        Performs subtraction with another Vector3D. Reference: Vector3D.Minus.
        @param other a Vector3D to be subtracted.
        @return a Vector3D that is the result of the subtraction.
        */
        template <class OtherType>
        auto operator-(const Vector3D<OtherType> &other) const;

        /*
        Performs inplace subtraction with another Vector3D.
        @param other a Vector3D to be subtracted.
        @return the reference of this Vector3D.
        */
        template <class OtherType>
        Vector3D<T> &operator-=(const Vector3D<OtherType> &other);

        /*
        Performs vector scaling.
        @param scaler a scaler used to scale this Vector3D.
        @return a Vector3D that is the result of the scaling.
        */
        template <class OtherType>
        auto Scale(const OtherType &scaler) const;

        /*
        Performs vector scaling. Reference: Vector3D.Scale.
        @param scaler a scaler used to scale this Vector3D.
        @return a Vector3D that is the result of the scaling.
        */
        template <class OtherType>
        auto operator*(const OtherType &scaler) const;

        /*
        Performs inplace vector scaling.
        @param scaler a scaler used to scale this Vector3D.
        @return the reference of this Vector3D.
        */
        template <class OtherType>
        Vector3D<T> &operator*=(const OtherType &scaler);

        /*
        Divides this Vector3D by a scaler.
        @param scaler a scaler used to divide this Vector3D.
        @return a Vector3D that is the result of the division.
        */
        template <class OtherType>
        auto Divide(const OtherType &scaler) const;

        /*
        Divides this Vector3D by a scaler. Reference: Vector3D.Divide
        @param scaler a scaler used to divide this Vector3D.
        @return a Vector3D that is the result of the division.
        */
        template <class OtherType>
        auto operator/(const OtherType &scaler) const;

        /*
        Performs inplace division on this Vector3D to be divided by a scaler.
        @param scaler a scaler used to divide this Vector3D.
        @return a reference of this Vector3D.
        */
        template <class OtherType>
        Vector3D<T> &operator/=(const OtherType &scaler);

        /*
        Performs dot product on this Vector3D with another Vector3D.
        @param other a Vector3D.
        @return a scaler that is the dot product.
        */
        template <class OtherType>
        decltype(auto) Dot(const Vector3D<OtherType> &other) const;

        /*
        Performs cross product on this Vector3D with another Vector3D.
        @param other a Vector3D.
        @return a Vector3D that is the cross product.
        */
        template <class OtherType>
        auto Cross(const Vector3D<OtherType> &other) const;

        /*
        Generates a new vector with normalized values of this Vector3D.
        @return a normalized vector.
        @thow DividedByZero when this Vector3D is a zero vector.
        */
        Vector3D<T> Normalized() const;

        /*
        Normalizes this Vector3D.
        @throw DividedByZero thown when this vector is a zero vector.
        */
        void Normalize();

        friend class std::ostream;
    };

}

#include "vector3d.tpp"

#endif
