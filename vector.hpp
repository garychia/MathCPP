#ifndef VECTOR_H
#define VECTOR_H

#include <initializer_list>
#include <array>
#include <omp.h>
#include <sstream>
#include <cmath>

#include "tuple.hpp"
#include "exceptions.hpp"

namespace DataStructure
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
        Constructor with Initial Size and a Value.
        @param s the initial size of the Vector to be generated.
        @param value the value the Vector will be filled with.
        */
        Vector(std::size_t s, const T &value) : Tuple<T>(s, value) {}

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Vector will store.
        */
        Vector(const std::initializer_list<T>& l) : Tuple<T>(l) {}

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the elements this Vector will store.
        */
        template <std::size_t N>
        Vector(const std::array<T, N>& arr) : Tuple<T>(arr) {}

        /*
        Copy Constructor
        @param other a Vector to be copied.
        */
        Vector(const Vector<T> &other) : Tuple<T>(other) {}

        /*
        Move Constructor
        @param other a Vector to be moved.
        */
        Vector(Vector &&other) : Tuple<T>(other) {}

        /*
        Copy Assignment
        @param other a Vector.
        @return a reference to this Vector.
        */
        virtual Vector<T> &operator=(const Vector<T> &other)
        {
            Tuple<T>::operator=(other);
            return *this;
        }

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        virtual T &operator[](const std::size_t &index)
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
        virtual const T &operator[](const std::size_t &index) const override
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
        Returns the length of this Vector.
        @return the length of this Vector.
        */
        template<class ReturnType>
        ReturnType Length() const
        {
            T squaredTotal = 0;
            for (std::size_t i = 0; i < this->size; i++)
            {
                T squaredElement = this->data[i] * this->data[i];
                squaredTotal += squaredElement;
            }
            return std::sqrt(squaredTotal);
        }

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