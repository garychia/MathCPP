#ifndef TUPLE_H
#define TUPLE_H

#include <sstream>
#include <vector>

#include "container.hpp"
#include "../Exceptions/exceptions.hpp"

namespace DataStructure
{
    template <class T>
    class Tuple : public Container<T>
    {
    public:
        /*
        Constructor that Generates an Empty Tuple.
        */
        Tuple() : Container<T>() {}

        /*
        Constructor with Initial Size and an Initial Value.
        @param s the initial size of the Tuple to be generated.
        @param value the value the Tuple will be filled with.
        */
        Tuple(std::size_t s, const T &value) : Container<T>(s, value) {}

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Tuple will store.
        */
        Tuple(const std::initializer_list<T>& l) : Container<T>(l) {}

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the elements this Tuple will store.
        */
        template <std::size_t N>
        Tuple(const std::array<T, N>& arr) : Container<T>(arr) { }

        /*
        Constructor with a std::vector.
        @param values a std::vector that contains the elements this Tuple
        will store.
        */
        Tuple(const std::vector<T> &values) : Container<T>(values) {}

        /*
        Copy Constructor
        @param other a Tuple to be copied.
        */
        Tuple(const Tuple<T> &other) : Container<T>(other) {}

        /*
        Copy Constructor
        @param other a Tuple to be copied.
        */
        template <class OtherType>
        Tuple(const Tuple<OtherType> &other) : Container<T>(other) {}

        /*
        Move Constructor
        @param other a Tuple to be moved.
        */
        Tuple(Tuple<T> &&other) : Container<T>(other) {}

        /*
        Move Constructor
        @param other a Tuple to be moved.
        */
        template<class OtherType>
        Tuple(Tuple<OtherType> &&other) : Container<T>(other) {}

        /*
        Access the element of a given index.
        @param index the index of the element to be accessed.
        @return the reference of the element accessed.
        @throw IndexOutOfBound when the index exceeds the largest possible index.
        */
        virtual const T &operator[](const std::size_t &index) const override
        {
            if (index < this->size)
                return this->data[index];
            else
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Tuple: Index must be non-negative and less than the number of elements.");
        }

        /*
        Copy Assignment
        @param other a Tuple to be copied.
        @return a reference to this Tuple.
        */
        virtual Tuple<T> &operator=(const Tuple<T> &other)
        {
            Container<T>::operator=(other);
            return *this;
        }

        /*
        Copy Assignment
        @param other a Tuple containing values of a different type
        to be copied.
        @return a reference to this Tuple.
        */
        template <class OtherType>
        Tuple<T> &operator=(const Tuple<OtherType> &other)
        {
            Container<T>::operator=(other);
            return *this;
        }

        /*
        Given two tuples, check if they have the same elements.
        @return bool that indicates whether the two tuples have the same
        elements.
        */
        template <class OtherType>
        bool operator==(const Tuple<OtherType> &other) const
        {
            if (this->Size() != other.Size())
                return false;
            for (std::size_t i = 0; i < this->Size(); i++)
                if ((*this)[i] != other[i])
                    return false;
            return true;
        }

        /*
        Given two tuples, check if they have different elements.
        @return bool that indicates whether the two tuples have different
        elements.
        */
        template <class OtherType>
        bool operator!=(const Tuple<OtherType> &other) const
        {
            return !operator==(other);
        }

        /*
        Converts this container to a string that shows the elements
        of this Container.
        @return a string that represents this Container.
        */
        virtual std::string ToString() const override
        {
            if (this->size == 0)
                return "(EMPTY)";
            std::stringstream ss;
            ss << "(";
            for (std::size_t i = 0; i < this->size; i++)
            {
                ss << this->data[i];
                if (i < this->size - 1)
                    ss << ", ";
            }
            ss << ")";
            return ss.str();
        }
        
        template <class OtherType>
        friend class Tuple;
    };
} // namespace Math

#endif