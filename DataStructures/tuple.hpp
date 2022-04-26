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
        Tuple();

        /*
        Constructor with Initial Size and an Initial Value.
        @param s the initial size of the Tuple to be generated.
        @param value the value the Tuple will be filled with.
        */
        Tuple(std::size_t s, const T &value);

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Tuple will store.
        */
        Tuple(const std::initializer_list<T>& l);

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the elements this Tuple will store.
        */
        template <std::size_t N>
        Tuple(const std::array<T, N>& arr);

        /*
        Constructor with a std::vector.
        @param values a std::vector that contains the elements this Tuple
        will store.
        */
        Tuple(const std::vector<T> &values);

        /*
        Copy Constructor
        @param other a Tuple to be copied.
        */
        Tuple(const Tuple<T> &other);

        /*
        Copy Constructor
        @param other a Tuple to be copied.
        */
        template <class OtherType>
        Tuple(const Tuple<OtherType> &other);

        /*
        Move Constructor
        @param other a Tuple to be moved.
        */
        Tuple(Tuple<T> &&other);

        /*
        Move Constructor
        @param other a Tuple to be moved.
        */
        template<class OtherType>
        Tuple(Tuple<OtherType> &&other);

        /*
        Access the element of a given index.
        @param index the index of the element to be accessed.
        @return the reference of the element accessed.
        @throw IndexOutOfBound when the index exceeds the largest possible index.
        */
        virtual const T &operator[](const std::size_t &index) const override;

        /*
        Copy Assignment
        @param other a Tuple to be copied.
        @return a reference to this Tuple.
        */
        virtual Tuple<T> &operator=(const Tuple<T> &other);

        /*
        Copy Assignment
        @param other a Tuple containing values of a different type
        to be copied.
        @return a reference to this Tuple.
        */
        template <class OtherType>
        Tuple<T> &operator=(const Tuple<OtherType> &other);

        /*
        Given two tuples, check if they have the same elements.
        @return bool that indicates whether the two tuples have the same
        elements.
        */
        template <class OtherType>
        bool operator==(const Tuple<OtherType> &other) const;

        /*
        Given two tuples, check if they have different elements.
        @return bool that indicates whether the two tuples have different
        elements.
        */
        template <class OtherType>
        bool operator!=(const Tuple<OtherType> &other) const;

        /*
        Converts this container to a string that shows the elements
        of this Container.
        @return a string that represents this Container.
        */
        virtual std::string ToString() const override;
        
        template <class OtherType>
        friend class Tuple;
    };
} // namespace Math

#include "tuple.tpp"

#endif