#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <cmath>
#include <sstream>
#include <ostream>

#include "container.hpp"
#include "exceptions.hpp"

namespace Math
{
    /*
    Vector3D is a class that represents a vector in a 3D space.
    */
    template <class T>
    class Vector3D : public Container<T>
    {
    public:
        // the X component of this vector
        T X;
        // the Y component of this vector
        T Y;
        // the Z component of this vector
        T Z;

        /*
        Constructor with Component X, Y and Z
        @param x the X component of this vector
        @param y the y component of this vector
        @param z the z component of this vector
        */
        Vector3D(T x = 0, T y = 0, T z = 0) : X(x), Y(y), Z(z) {}

        /*
        Copy Constructor
        @param other a Vector3D to be copied.
        */
        template <class OtherType>
        Vector3D(const Vector3D<OtherType> &other) : X(other.X), Y(other.Y), Z(other.Z) {}

        /*
        Returns a string that represents this Vector3D.
        @return a string that contains information about this Vector3D.
        */
        auto ToString() const
        {
            std::stringstream ss;
            ss << "(" << X << ", " << Y << ", " << Z << ")";
            return ss.str();
        }

        /*
        Returns a component of this Vector3D.
        @param index the index of the X, Y or Z component.
        @return X, Y or Z component when index is 0, 1 or 2, respectively.
        @throw IndexOutOfBound an exception thrown when index is not valid.
        */
        virtual T &operator[](const std::size_t &index) override
        {
            switch (index)
            {
            case 0:
                return X;
            case 1:
                return Y;
            case 2:
                return Z;
            default:
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Vector3D: Index must be between 0 and 2 inclusively.");
            }
        }

        /*
        Returns the number of elements this Vector3D has.
        @return always 3 (X, Y and Z components).
        */
        virtual std::size_t Size() const override { return 3; }

        /*
        Calculates the length of this Vector3D.
        @return the length of this Vector3D.
        */
        T Length() const { return std::sqrt(X * X + Y * Y + Z * Z); }

        /*
        Performs addition with another Vector3D.
        @param other a Vector3D to be added.
        @return a Vector3D that is the result of the addition.
        */
        template <class OtherType>
        auto Add(const Vector3D<OtherType> &other) const
        {
            return Vector3D<decltype(X + other.X)>(X + other.X, Y + other.Y, Z + other.Z);
        }

        /*
        Performs addition with another Vector3D. Reference: Vector3D.Add.
        @param other a Vector3D to be added.
        @return a Vector3D that is the result of the addition.
        */
        template <class OtherType>
        auto operator+(const Vector3D<OtherType> &other) const
        {
            return this->Add(other);
        }

        /*
        Performs inplace addition with another Vector3D..
        @param other a Vector3D to be added.
        @return the reference of this Vector3D.
        */
        template <class OtherType>
        Vector3D<T> &operator+=(const Vector3D<OtherType> &other)
        {
            X += other.X;
            Y += other.Y;
            Z += other.Z;
            return *this;
        }

        /*
        Performs subtraction with another Vector3D.
        @param other a Vector3D to be subtracted.
        @return a Vector3D that is the result of the subtraction.
        */
        template <class OtherType>
        auto Minus(const Vector3D<OtherType> &other) const
        {
            return Vector3D<decltype(X - other.X)>(X - other.X, Y - other.Y, Z - other.Z);
        }

        /*
        Performs subtraction with another Vector3D. Reference: Vector3D.Minus.
        @param other a Vector3D to be subtracted.
        @return a Vector3D that is the result of the subtraction.
        */
        template <class OtherType>
        auto operator-(const Vector3D<OtherType> &other) const
        {
            return this->Minus(other);
        }

        /*
        Performs inplace subtraction with another Vector3D.
        @param other a Vector3D to be subtracted.
        @return the reference of this Vector3D.
        */
        template <class OtherType>
        Vector3D<T> &operator-=(const Vector3D<OtherType> &other)
        {
            X -= other.X;
            Y -= other.Y;
            Z -= other.Z;
            return *this;
        }

        /*
        Performs vector scaling.
        @param scaler a scaler used to scale this Vector3D.
        @return a Vector3D that is the result of the scaling.
        */
        template <class OtherType>
        auto Scale(const OtherType &scaler) const
        {
            return Vector3D<decltype(X * scaler)>(X * scaler, Y * scaler, Z * scaler);
        }

        /*
        Performs vector scaling. Reference: Vector3D.Scale.
        @param scaler a scaler used to scale this Vector3D.
        @return a Vector3D that is the result of the scaling.
        */
        template <class OtherType>
        auto operator*(const OtherType &scaler) const
        {
            return this->Scale(scaler);
        }

        /*
        Performs inplace vector scaling.
        @param scaler a scaler used to scale this Vector3D.
        @return the reference of this Vector3D.
        */
        template <class OtherType>
        Vector3D<T> &operator*=(const OtherType &scaler)
        {
            X *= scaler;
            Y *= scaler;
            Z *= scaler;
            return *this;
        }


        /*
        Divides this Vector3D by a scaler.
        @param scaler a scaler used to divide this Vector3D.
        @return a Vector3D that is the result of the division.
        */
        template <class OtherType>
        auto Divide(const OtherType &scaler) const
        {
            return Vector3D<decltype(X / scaler)>(X / scaler, Y / scaler, Z / scaler);
        }

        /*
        Divides this Vector3D by a scaler. Reference: Vector3D.Divide
        @param scaler a scaler used to divide this Vector3D.
        @return a Vector3D that is the result of the division.
        */
        template <class OtherType>
        auto operator/(const OtherType &scaler) const
        {
            return this->Divide(scaler);
        }

        /*
        Performs inplace division on this Vector3D to be divided by a scaler.
        @param scaler a scaler used to divide this Vector3D.
        @return a reference of this Vector3D.
        */
        template <class OtherType>
        Vector3D<T> &operator/=(const OtherType &scaler)
        {
            X /= scaler;
            Y /= scaler;
            Z /= scaler;
            return *this;
        }

        /*
        Performs dot product on this Vector3D with another Vector3D.
        @param other a Vector3D.
        @return a scaler that is the dot product.
        */
        template <class OtherType>
        decltype(auto) Dot(const Vector3D<OtherType> &other) const
        {
            return X * other.X + Y * other.Y + Z * other.Z;
        }

        /*
        Performs cross product on this Vector3D with another Vector3D.
        @param other a Vector3D.
        @return a Vector3D that is the cross product.
        */
        template <class OtherType>
        auto Cross(const Vector3D<OtherType> &other) const
        {
            return Vector3D<decltype(Y * other.Z)>(
                Y * other.Z - other.Y * Z,
                Z * other.X - other.Z * X,
                X * other.Y - other.X * Y);
        }

        /*
        Generates a new vector with normalized values of this Vector3D.
        @return a normalized vector.
        @thow DividedByZero when this Vector3D is a zero vector.
        */
        Vector3D<T> Normalized() const
        {
            const T length = Length();
            if (length == 0)
                throw Exceptions::DividedByZero("Cannot normalize a zero vector.")
            return Vector3D<T>(X / length, Y / length, Z / length);
        }

        /*
        Normalizes this Vector3D.
        @throw DividedByZero thown when this vector is a zero vector.
        */
        void Normalize()
        {
            const T length = Length();
            if (length == 0)
                throw Exceptions::DividedByZero("Cannot normalize a zero vector.")
            X /= length;
            Y /= length;
            Z /= length;
        }

        /*
        Converts this Vector3D to a string and pass it to an output stream.
        @param stream an output stream.
        @param v a Vector3D
        @return a reference to the output stream.
        */
        friend std::ostream &operator<<(std::ostream &stream, const Vector3D<T> &v)
        {
            stream << v.ToString();
            return stream;
        }
    };

}

#endif
