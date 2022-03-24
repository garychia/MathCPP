#ifndef VECTOR_H
#define VECTOR_H

#include <initializer_list>
#include <array>
#include <omp.h>

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
        /* Constructor with no elements. */
        Vector() : dimension(0), data(nullptr) {}

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Vector will store.
        */
        Vector(std::initializer_list<T> l) : dimension(l.size())
        {
            data = new T[l.size()];
            #pragma omp parallel for schedule(dynamic, 4)
            for (std::size_t i = 0; i < l.size(); i++)
                data[i] = *(l.begin() + i);
        }

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the elements this Vector will store.
        */
       template <std::size_t N>
        Vector(const std::array<T, N>& arr) : dimension(arr.size())
        {
            data = new T[arr.size()];
            #pragma omp parallel for schedule(dynamic, 4)
            for (std::size_t i = 0; i < arr.size(); i++)
                data[i] = arr[i];
        }

        /*
        Copy Constructor
        @param other a Vector to be copied.
        */
        Vector(const Vector<T> &other)
        {
            T *newData = new T[other.dimension];
            #pragma omp parallel for schedule(dynamic, 4)
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
            if (data)
                delete[] data;
        }

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        T &operator[](const std::size_t &index) override
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
        virtual std::size_t Size() const override { return dimension; }

        /*
        Generates a vector filled with zeros.
        @param n the number of zeros.
        @return a Vector that is filled with n zeros.
        */
        static Vector<T> ZeroVector(const std::size_t& n)
        {
            Vector<T> v;
            v.dimension = n;
            v.data = new T[n];
            #pragma omp parallel for schedule(dynamic, 4)
            for (std::size_t i = 0; i < n; i++)
                v.data[i] = 0;
            return v;
        }

        friend class Matrix;

    private:
        // the dimension of this Vector
        std::size_t dimension;
        // an array that contians the elements
        T *data;
    };
}

#endif