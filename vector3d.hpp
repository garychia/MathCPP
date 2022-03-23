#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <sstream>
#include <ostream>

#include "container.hpp"
#include "exceptions.hpp"

namespace Math
{
    template <class T>
    class Vector3D : public Container<T>
    {
    public:
        T X;
        T Y;
        T Z;

        Vector3D(T x = 0, T y = 0, T z = 0) : X(x), Y(y), Z(z) {}

        template <class OtherType>
        Vector3D(const Vector3D<OtherType> &other) : X(other.X), Y(other.Y), Z(other.Z) {}

        auto ToString() const
        {
            std::stringstream ss;
            ss << "(" << X << ", " << Y << ", " << Z << ")";
            return ss.str();
        }

        virtual T &operator[](const std::size_t &index)
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

        virtual std::size_t Size() const { return 3; }

        template <class OtherType>
        auto Add(const Vector3D<OtherType> &other) const
        {
            return Vector3D<decltype(X + other.X)>(X + other.X, Y + other.Y, Z + other.Z);
        }

        template <class OtherType>
        auto operator+(const Vector3D<OtherType> &other) const
        {
            return this->Add(other);
        }

        template <class OtherType>
        Vector3D<T> &operator+=(const Vector3D<OtherType> &other)
        {
            X += other.X;
            Y += other.Y;
            Z += other.Z;
            return *this;
        }

        template <class OtherType>
        auto Minus(const Vector3D<OtherType> &other) const
        {
            return Vector3D<decltype(X - other.X)>(X - other.X, Y - other.Y, Z - other.Z);
        }

        template <class OtherType>
        auto operator-(const Vector3D<OtherType> &other) const
        {
            return this->Minus(other);
        }

        template <class OtherType>
        Vector3D<T> &operator-=(const Vector3D<OtherType> &other)
        {
            X -= other.X;
            Y -= other.Y;
            Z -= other.Z;
            return *this;
        }

        template <class OtherType>
        auto Scale(const OtherType &scaler) const
        {
            return Vector3D<decltype(X * scaler)>(X * scaler, Y * scaler, Z * scaler);
        }

        template <class OtherType>
        auto operator*(const OtherType &scaler) const
        {
            return this->Scale(scaler);
        }

        template <class OtherType>
        Vector3D<T> &operator*=(const OtherType &scaler)
        {
            X *= scaler;
            Y *= scaler;
            Z *= scaler;
            return *this;
        }

        template <class OtherType>
        auto Divide(const OtherType &scaler) const
        {
            return Vector3D<decltype(X / scaler)>(X / scaler, Y / scaler, Z / scaler);
        }

        template <class OtherType>
        auto operator/(const OtherType &scaler) const
        {
            return this->Divide(scaler);
        }

        template <class OtherType>
        Vector3D<T> &operator/=(const OtherType &scaler)
        {
            X /= scaler;
            Y /= scaler;
            Z /= scaler;
            return *this;
        }

        template <class OtherType>
        auto Dot(const Vector3D<OtherType> &other) const
        {
            return Vector3D<decltype(X * other.X)>(X * other.X, Y * other.Y, Z * other.Z);
        }

        template <class OtherType>
        auto Cross(const Vector3D<OtherType> &other) const
        {
            return Vector3D<decltype(Y * other.Z)>(
                Y * other.Z - other.Y * Z,
                Z * other.X - other.Z * X,
                X * other.Y - other.X * Y);
        }

        friend std::ostream &operator<<(std::ostream &stream, const Vector3D<T> &v)
        {
            stream << v.ToString();
            return stream;
        }
    };

}

#endif
