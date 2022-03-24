#ifndef VECTOR_H
#define VECTOR_H

#include <initializer_list>
#include <array>
#include <omp.h>
#include <sstream>

#include "tuple.hpp"
#include "exceptions.hpp"

namespace Math
{
    /*
    Vector is a Container that is capable of storing multiple elements
    such as numbers.
    */
    template <class T>
    class Vector : public Tuple<T>
    {
    public:
        /*
        Constructor that Generates an Empty Vector.
        */
        Vector() : Tuple<T>() {}

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Vector will store.
        */
        Vector(std::initializer_list<T> l) : Tuple<T>(l) {}

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the elements this Vector will store.
        */
        template <std::size_t N>
        Vector(const std::array<T, N>& arr) : Tuple<T>(arr) {}

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        T &operator[](const std::size_t &index)
        {
            if (index > this->size - 1)
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Vector: Index must be less than the dimention.");
            return this->data[index];
        }

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        const T &operator[](const std::size_t &index) const override
        {
            if (index > this->size - 1)
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Vector: Index must be less than the dimention.");
            return this->data[index];
        }

        /*
        Returns the dimention of this Vector.
        @return the dimention of this Vector.
        */
        std::size_t Dimension() const { return this->size; }

        /*
        Generates a vector filled with zeros.
        @param n the number of zeros.
        @return a Vector that is filled with n zeros.
        */
        static Vector<T> ZeroVector(const std::size_t& n)
        {
            Vector<T> v;
            v.size = n;
            v.data = new T[n];
            #pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < n; i++)
                v.data[i] = 0;
            return v;
        }
    };
}

#endif