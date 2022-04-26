#ifndef LIST_H
#define LIST_H

#define INITIAL_SIZE 4

#include "container.hpp"
#include "../Exceptions/exceptions.hpp"

namespace DataStructure
{
    /*
    List is an ordered collection of elements.
    */
    template <class T>
    class List : public Container<T>
    {
    private:
        // The number of elements this List contains.
        std::size_t nElements;

        // Resize the list.
        void resize();

        // Shrink the list.
        void shrink();

    public:
        /*
        Constructor that Generates an Empty List.
        */
        List();

        /*
        Constructor with Initial Size and a Value.
        @param s the initial size of the List to be generated.
        @param value the value the List will be filled with.
        */
        List(std::size_t s, const T &value);

        /*
        Constructor with Initializer List as Input.
        @param l a std::initializer_list that contains the elements this List will store.
        */
        List(const std::initializer_list<T> &l);

        /*
        Constructor with an Arrary as Input.
        @param arr a std::array that contains the elements this List will store.
        */
        template <std::size_t N>
        List(const std::array<T, N> &arr);

        /*
        Copy Constructor
        @param other a List to be copied.
        */
        List(const List<T> &other);

        /*
        Move Constructor
        @param other a Vector to be moved.
        */
        List(List &&other);

        /*
        Copy Assignment
        @param other a List.
        @return a reference to this List.
        */
        virtual List<T> &operator=(const List<T> &other);

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        virtual T &operator[](const std::size_t &index);

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        virtual const T &operator[](const std::size_t &index) const override;

        /*
        Returns the number of elements this List stores.
        @return the number of elements this List stores.
        */
        virtual std::size_t Size() const override;

        /*
        Checks if this List is empty or not.
        @return a bool that indicates whether this List is empty.
        */
        virtual bool IsEmpty() const override;

        /*
        Converts this List to a string that displays all the elements
        of this List.
        @return a string that represents this List.
        */
        virtual std::string ToString() const override;

        /*
        Appends an element to this List.
        @param element the element to be appended.
        */
        virtual void Append(const T &element);

        /*
        Appends an element to this List.
        @param element the element to be appended (and moved).
        */
        virtual void Append(T &&element);

        /*
        Prepends an element to this List.
        @param element the element to be prepended.
        */
        virtual void Prepend(const T &element);

        /*
        Prepends an element to this List.
        @param element the element to be prepended (and moved).
        */
        virtual void Prepend(T &&element);

        /*
        Pops the last element from this List.
        @return the popped element.
        @throw EmptyList when the list is already empty.
        */
        virtual T PopEnd();

        /*
        Pops the first element from this List.
        @return the popped element.
        @throw EmptyList when the list is already empty.
        */
        virtual T PopFront();

        /*
        Clears all the elements this List stores.
        */
        void Clear();

        template <class OtherType>
        friend class List;
    };
} // namespace DataStructure

#include "list.tpp"

#endif