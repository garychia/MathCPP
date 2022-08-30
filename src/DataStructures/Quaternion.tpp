#include "Exceptions.hpp"

namespace DataStructures {
    template <class T>
    Quaternion<T>::Quaternion() : components({0, 1, 0, 0}) {}

    template <class T> Quaternion<T>::Quaternion(Vector<T> &axis, T angle) {
    if (axis.Dimension() != 3)
        throw Exceptions::InvalidArgument(
            "Quaternion: Expected the demension of axis to be 3.");
        const auto halfAngle = angle / 2;
        const auto sinHalfAngle = Math::Sine(halfAngle);
        const auto cosHalfAngle = Math::Cosine(halfAngle);
        components =
        {
            axis[0] * sinHalfAngle,
            axis[1] * sinHalfAngle,
            axis[2] * sinHalfAngle,
            cosHalfAngle
        };
    }

    template <class T>
    Quaternion<T>::Quaternion(T x, T y, T z, T angle)
    {
        const auto halfAngle = angle / 2;
        const auto sinHalfAngle = Math::Sine(halfAngle);
        const auto cosHalfAngle = Math::Cosine(halfAngle);
        components =
        {
            axis[0] * sinHalfAngle,
            axis[1] * sinHalfAngle,
            axis[2] * sinHalfAngle,
            cosHalfAngle
        };
    }    
} // namespace DataStructures