#ifndef MATH_H
#define MATH_H

#include "../DataStructures/vector.hpp"

using namespace DataStructure;

namespace Math
{
    template <class T>
    Vector<T> Power(const Vector<T>& v)
    {
        return v * v;
    }

    template <class T>
    Matrix<T> Power(const Matrix<T>& m)
    {
        return m.Scale(m);
    }
} // namespace Math

#endif
