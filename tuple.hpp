#ifndef TUPLE_H
#define TUPLE_H

#include <sstream>

#include "container.hpp"
#include "exceptions.hpp"

namespace Math
{
    template <class T>
    class Tuple : public Container<T>
    {
    protected:
        // number of elements stored
        std::size_t size;
        // array of the elements
        T *data;

    public:
        /*
        Constructor that Generates an Empty Tuple.
        */
        Tuple() : size(0), data(nullptr) {}

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Tuple will store.
        */
        Tuple(std::initializer_list<T> l) : size(l.size())
        {
            if (l.size() > 0)
            {
                data = new T[l.size()];
                #pragma omp parallel for schedule(dynamic)
                for (std::size_t i = 0; i < l.size(); i++)
                    data[i] = *(l.begin() + i);
            }
            else
                data = nullptr;
        }

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the elements this Tuple will store.
        */
       template <std::size_t N>
        Tuple(const std::array<T, N>& arr) : size(arr.size())
        {
            if (arr.size() > 0)
            {
                data = new T[arr.size()];
                #pragma omp parallel for schedule(dynamic)
                for (std::size_t i = 0; i < arr.size(); i++)
                    data[i] = arr[i];
            }
            else
                data = nullptr;
        }

        /*
        Copy Constructor
        @param other a Tuple to be copied.
        */
        Tuple(const Tuple<T> &other)
        {
            size = other.size;
            if (size > 0)
            {
                T *newData = new T[size];
                #pragma omp parallel for schedule(dynamic)
                for (std::size_t i = 0; i < size; i++)
                    newData[i] = other.data[i];
                delete[] data;
                data = newData;
            }
            else
                data = nullptr;
        }

        /*
        Move Constructor
        @param other a Tuple to be moved.
        */
        Tuple(Tuple &&other)
        {
            size = move(other.size);
            data = move(other.data);
            other.size = 0;
            other.data = nullptr;
        }

        /*
        Destructor
        */
        virtual ~Tuple()
        {
            if (data)
                delete[] data;
        }

        /*
        Access the element of a given index.
        @param index the index of the element to be accessed.
        @return the reference of the element accessed.
        @throw IndexOutOfBound when the index exceeds the largest possible index.
        */
        virtual const T &operator[](const std::size_t &index) const override
        {
            if (index < size)
                return data[index];
            else
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Tuple: Index must be non-negative and less than the number of elements.");
        }

        /*
        Copy Assignment
        @param other a Tuple.
        @return a reference to this Tuple.
        */
        virtual Tuple<T> &operator=(const Tuple<T> &other) {
            if (this != &other)
            {
                size = other.size;
                delete[] data;
                data = nullptr;
                if (size > 0) {
                    data = new T[size];
                    #pragma omp parallel for schedule(dynamic)
                    for (std::size_t i = 0; i < size; i++)
                        data[i] = other.data[i];
                }
            }
            return *this;
        }

        /*
        Returns the number of elements this Tuple stores.
        @return the number of elements this Tuple stores.
        */
        virtual std::size_t Size() const override { return size; }

        /*
        Converts this container to a string that shows the elements
        of this Container.
        @return a string that represents this Container.
        */
        virtual std::string ToString() const override
        {
            if (size == 0)
                return "(EMPTY)";
            std::stringstream ss;
            ss << "(";
            for (std::size_t i = 0; i < size; i++)
            {
                ss << data[i];
                if (i < size - 1)
                    ss << ", ";
            }
            ss << ")";
            return ss.str();
        }

        /*
        Converts this Tuple to a string and pass it to an output stream.
        @param stream an output stream.
        @param t a Tuple
        @return a reference to the output stream.
        */
        friend std::ostream &operator<<(std::ostream &stream, const Tuple<T> &t)
        {
            stream << t.ToString();
            return stream;
        }
    };
} // namespace Math

#endif