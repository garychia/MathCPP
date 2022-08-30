#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include "Vector.hpp"
#include "Math.hpp"

namespace DataStructures
{
    template <class T>
    class Quaternion
    {
        Vector<T> components;

        Quaternion();

        Quaternion(Vector<T> &axis, T angle = 0);

        Quaternion(T x = 0, T y = 0, T z = 0, T angle = 0);
    };

}

#include "Quaternion.tpp"

#endif