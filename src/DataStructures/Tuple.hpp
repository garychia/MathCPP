#ifndef TUPLE_HPP
#define TUPLE_HPP

#include <vector>

#include "Container.hpp"

namespace DataStructures
{
    /* An immutable container of a fixed size. */
    template <class T>
    class Tuple : public Container<T>
    {
    public:
        /* Constructor that Generates an Empty Tuple. */
        Tuple();

        /**
         * Constructor that generates a Tuple filled with identical values.
         * @param s the size (number of elements) of the Tuple.
         * @param value the value the Tuple will be filled with.
         **/
        Tuple(std::size_t s, const T &value);

        /**
         * Constructor with a std::initializer_list as Input.
         * @param l a std::initializer_list that contains the elements this Tuple will store.
         **/
        Tuple(const std::initializer_list<T> &l);

        /**
         * Constructor with a std::arrary as Input.
         * @param arr a std::array that contains the elements this Tuple will store.
         **/
        template <std::size_t N>
        Tuple(const std::array<T, N> &arr);

        /**
         * Constructor with a std::vector as Input.
         * @param values a std::vector that contains the elements this Tuple will store.
         **/
        Tuple(const std::vector<T> &values);

        /**
         * Copy Constructor.
         * @param other a Container whose elements the Tuple will copy.
         **/
        Tuple(const Container<T> &other);

        /**
         * Copy Constructor with a Container whose elements are of a different type.
         * @param other a Container whose elements the Tuple will copy.
         **/
        template <class OtherType>
        Tuple(const Container<OtherType> &other);

        /**
         * Move Constructor.
         * @param other a Container whose elements will be 'moved' into the Tuple.
         **/
        Tuple(Container<T> &&other);

        /**
         * Access the element at a given index.
         * @param index the index at which the element will be accessed.
         * @return the reference to the element accessed.
         * @throw IndexOutOfBound when the index exceeds the largest possible index.
         **/
        virtual const T &operator[](const std::size_t &index) const override;

        /**
         * Copy Assignment.
         * @param other a Container whose elements will be copied into the Tuple.
         * @return a reference to the Tuple.
         **/
        virtual Tuple<T> &operator=(const Container<T> &other) override;

        /**
         * Copy Assignment
         * @param other a Container containing elements of a different type to be copied.
         * @return a reference to the Tuple.
         **/
        template <class OtherType>
        Tuple<T> &operator=(const Container<OtherType> &other);

        /**
         * Check if two Tuples have the same elements.
         * @param other a Tuple to be compared with this Tuple.
         * @return a bool that indicates whether both Tuples have the same elements.
         **/
        template <class OtherType>
        bool operator==(const Tuple<OtherType> &other) const;

        /**
         * Check if two Tuples have different elements.
         * @param other a Tuple to be compared with this Tuple.
         * @return bool that indicates whether both Tuples have different elements.
         **/
        template <class OtherType>
        bool operator!=(const Tuple<OtherType> &other) const;

        /**
         * Generate a string that displays the elements of the Tuple.
         * @return a string that displays the elements of the Tuple.
         **/
        virtual std::string ToString() const override;

        template <class OtherType>
        friend class Tuple;
    };
} // namespace DataStructures

#include "Tuple.tpp"

#endif