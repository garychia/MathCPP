#ifndef CONTAINER_HPP
#define CONTAINER_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <iostream>

namespace DataStructure
{
    /*
    Container is an abstract class that is capable of storing data.
    */
    template <class T>
    class Container
    {
    protected:
        // number of elements stored
        std::size_t size;
        // array of the elements
        T *data;

    public:
        /*
        Constructor that Generates an Empty Container.
        */
        Container();

        /*
        Constructor with Initial Size and an Initial Value.
        @param s the initial size of the Container to be generated.
        @param value the value the Container will be filled with.
        */
        Container(std::size_t s, const T &value);

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Container
        will store.
        */
        Container(const std::initializer_list<T> &l);

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the elements this Container will store.
        */
        template <std::size_t N>
        Container(const std::array<T, N> &arr);

        /*
        Constructor with a std::vector.
        @param values a std::vector that contains the elements this Container
        will store.
        */
        Container(const std::vector<T> &values);

        /*
        Copy Constructor
        @param other a Container to be copied.
        */
        Container(const Container<T> &other);

        /*
        Copy Constructor
        @param other a Container to be copied.
        */
        template <class OtherType>
        Container(const Container<OtherType> &other);

        /*
        Move Constructor
        @param other a Container to be moved.
        */
        Container(Container<T> &&other);

        /*
        Move Constructor
        @param other a Container to be moved.
        */
        template <class OtherType>
        Container(Container<OtherType> &&other);

        /*
        Destructor
        */
        virtual ~Container();

        /*
        Access the element of a given index.
        @param index the index of the element to be accessed.
        @return the reference of the element accessed.
        */
        virtual const T &operator[](const std::size_t &index) const = 0;

        /*
        Copy Assignment
        @param other a Container to be copied.
        @return a reference to this Container.
        */
        virtual Container<T> &operator=(const Container<T> &other);

        /*
        Copy Assignment
        @param other a Container containing values of a different type to be copied.
        @return a reference to this Container.
        */
        template <class OtherType>
        Container<T> &operator=(const Container<OtherType> &other);

        /*
        Returns the number of elements this Container stores.
        @return the number of elements this Container stores.
        */
        virtual std::size_t Size() const;

        /*
        Checks if this Container is empty or not.
        @return a bool that indicates whether this Container is empty.
        */
        virtual bool IsEmpty() const;

        /*
        Converts this container to a string that shows the elements
        of this Container.
        @return a string that represents this Container.
        */
        virtual std::string ToString() const = 0;

        friend std::ostream;
    };
} // namespace DataStructure

#include "Container.tpp"

#endif