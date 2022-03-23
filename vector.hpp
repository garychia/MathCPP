#ifndef VECTOR_H
#define VECTOR_H

#include <initializer_list>

#include "container.hpp"
#include "exceptions.hpp"

namespace Math
{
    template <class T>
    class Vector : public Container<T>
    {
    public:
        Vector(std::initializer_list<T> l) : nDimension(l.size())
        {
            data = new T[l.size()];
            for (std::size_t i = 0; i < l.size(); i++)
                data[i] = *(l.begin() + i);
        }

        Vector(const Vector<T> &other)
        {
            T *newData = new T[other.nDimension];
            for (std::size_t i = 0; i < other.nDimension; i++)
                newData[i] = other.data[i];
            delete[] data;
            data = newData;
        }

        Vector(Vector<T> &&other)
        {
            nDimension = other.nDimension;
            data = other.data;
            other.nDimension = 0;
            other.data = nullptr;
        }

        ~Vector()
        {
            delete[] data;
        }

        T &operator[](const std::size_t &index)
        {
            if (index > nDimension - 1)
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Vector: Index must be less than the dimention.");
            return data[index];
        }

        std::size_t Dimension() const { return nDimension; }

        std::size_t Size() const { return nDimension; }

    private:
        std::size_t nDimension;
        T *data;
    };
}

#endif