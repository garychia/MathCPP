#ifndef LIST_H
#define LIST_H

#define INITIAL_SIZE 4

#include "container.hpp"
#include "exceptions.hpp"

namespace DataStructure
{
    template <class T>
    class List : public Vector<T>
    {
    private:
        std::size_t nElements;

        void resize()
        {
            this->size = this->size > 2 ? this->size << 1 : INITIAL_SIZE;
            T *newData = new T[this->size];
            for (std::size_t i = 0; i < nElements; i++)
                newData[i] = this->data[i];
            if (this->data)
                delete[] this->data;
            this->data = newData;
        }

        void shrink()
        {
            this->size = this->size / 2 > INITIAL_SIZE ? this->size / 2 : INITIAL_SIZE;
            T *newData = new T[this->size];
            for (std::size_t i = 0; i < nElements; i++)
                newData[i] = this->data[i];
            this->data = newData;
        }

    public:
        /*
        Constructor that Generates an Empty List.
        */
        List() : Vector<T>(), nElements(0)
        {
            this->data = new T[INITIAL_SIZE];
            this->size = INITIAL_SIZE;
        }

        /*
        Constructor with Initializer List as Input.
        @param l a std::initializer_list that contains the elements this List will store.
        */
        List(const std::initializer_list<T> &l) : Vector<T>(l)
        {
            nElements = this->size;
        }

        /*
        Constructor with an Arrary as Input.
        @param arr a std::array that contains the elements this List will store.
        */
        template <std::size_t N>
        List(const std::array<T, N> &arr) : Vector<T>(arr)
        {
            nElements = this->size;
        }

        /*
        Copy Constructor
        @param other a List to be copied.
        */
        List(const List<T> &other) : Vector<T>(other)
        {
            nElements = this->size;
        }

        /*
        Move Constructor
        @param other a Vector to be moved.
        */
        List(List &&other) : Vector<T>(other)
        {
            nElements = this->size;
        }

        /*
        Copy Assignment
        @param other a List.
        @return a reference to this List.
        */
        virtual List<T> &operator=(const List<T> &other)
        {
            Vector<T>::operator=(other);
            nElements = this->size;
            return *this;
        }

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        virtual T &operator[](const std::size_t &index) override
        {
            if (index > nElements - 1)
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
            if (index > nElements - 1)
                throw Exceptions::IndexOutOfBound(
                    index,
                    "List: Index must be less than the number of elements.");
            return this->data[index];
        }

        /*
        Returns the number of elements this List stores.
        @return the number of elements this List stores.
        */
        virtual std::size_t Size() const override { return nElements; }

        /*
        Appends an element to this List.
        @param element the element to be appended.
        */
        virtual void Append(const T &element)
        {
            if (nElements + 1 > this->size)
                resize();
            this->data[nElements++] = element;
        }

        /*
        Appends an element to this List.
        @param element the element to be appended (and moved).
        */
        virtual void Append(T &&element)
        {
            if (nElements + 1 > this->size)
                resize();
            this->data[nElements++] = std::move(element);
        }

        /*
        Prepends an element to this List.
        @param element the element to be prepended.
        */
        virtual void Prepend(const T &element)
        {
            if (nElements + 1 > this->size)
                resize();
            nElements++;
            for (std::size_t i = nElements - 1; i > 0; i--)
                this->data[i] = this->data[i - 1];
            this->data[0] = element;
        }

        /*
        Prepends an element to this List.
        @param element the element to be prepended (and moved).
        */
        virtual void Prepend(T &&element)
        {
            if (nElements + 1 > this->size)
                resize();
            nElements++;
            for (std::size_t i = nElements - 1; i > 0; i--)
                this->data[i] = this->data[i - 1];
            this->data[0] = std::move(element);
        }

        /*
        Pops the last element from this List.
        @return the popped element.
        @throw EmptyList when the list is already empty.
        */
        virtual T PopEnd()
        {
            if (nElements == 0)
                throw Exceptions::EmptyList("List: Trying to pop an element from an empty list.");
            T element = this->data[--nElements];
            if (nElements < this->size / 2)
                shrink();
            return element;
        }

        /*
        Pops the first element from this List.
        @return the popped element.
        @throw EmptyList when the list is already empty.
        */
        virtual T PopFront()
        {
            if (nElements == 0)
                throw Exceptions::EmptyList("List: Trying to pop an element from an empty list.");
            T element = this->data[0];
            nElements--;
            for (std::size_t i = 0; i < nElements; i++)
                this->data[i] = this->data[i + 1];
            if (nElements < this->size / 2)
                shrink();
            return element;
        }
    };
} // namespace DataStructure

#endif