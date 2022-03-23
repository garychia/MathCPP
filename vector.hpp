#ifndef VECTOR_H
#define VECTOR_H

#include <initializer_list>

#include "container.hpp"
#include "exceptions.hpp"

namespace Math
{
    /*
    Vector is a Container that is capable of storing multiple elements
    such as numbers.
    */
    template <class T>
    class Vector : public Container<T>
    {
    public:
        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Vector will store.
        */
        Vector(std::initializer_list<T> l) : dimension(l.size())
        {
            data = new T[l.size()];
            for (std::size_t i = 0; i < l.size(); i++)
                data[i] = *(l.begin() + i);
        }

        /*
        Copy Constructor
        @param other a Vector to be copied.
        */
        Vector(const Vector<T> &other)
        {
            T *newData = new T[other.dimension];
            for (std::size_t i = 0; i < other.dimension; i++)
                newData[i] = other.data[i];
            delete[] data;
            data = newData;
        }

        /*
        Move Constructor
        @param other a Vector to be moved.
        */
        Vector(Vector<T> &&other)
        {
            dimension = other.dimension;
            data = other.data;
            other.dimension = 0;
            other.data = nullptr;
        }

        /* Destructor */
        ~Vector()
        {
            delete[] data;
        }

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        T &operator[](const std::size_t &index)
        {
            if (index > dimension - 1)
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Vector: Index must be less than the dimention.");
            return data[index];
        }

        /*
        Returns the dimention of this Vector.
        @return the dimention of this Vector.
        */
        std::size_t Dimension() const { return dimension; }

        /*
        Returns the number of elements this Vector contains.
        @return the number of elements this Vector contains.
        */
        std::size_t Size() const { return dimension; }

    private:
        // the dimension of this Vector
        std::size_t dimension;
        // an array that contians the elements
        T *data;
    };
}

#endif