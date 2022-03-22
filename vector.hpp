#ifndef VECTOR_H
#define VECTOR_H

#include <initializer_list>

namespace Math
{
    template <class T>
    class Vector
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

        template<class IndexType>
        T &operator[](const IndexType &index)
        {
            return data[(std::size_t)index];
        }

        std::size_t Dimension() const { return nDimension; }

    private:
        std::size_t nDimension;
        T *data;
    };
}

#endif