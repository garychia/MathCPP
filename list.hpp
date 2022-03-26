#ifndef LIST_H
#define LIST_H

#define INITIAL_SIZE 4

#include "container.hpp"
#include "exceptions.hpp"

namespace DataStructure
{
    template <class T>
    class List : public Container<T>
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
            if (this->data)
                delete[] this->data;
            this->data = newData;
        }

    public:
        /*
        Constructor that Generates an Empty List.
        */
        List() : Container<T>(), nElements(0)
        {
            this->data = new T[INITIAL_SIZE];
            this->size = INITIAL_SIZE;
        }

        /*
        Constructor with Initial Size and a Value.
        @param s the initial size of the List to be generated.
        @param value the value the List will be filled with.
        */
        List(std::size_t s, const T &value) : Container<T>(s, value), nElements(s) {}

        /*
        Constructor with Initializer List as Input.
        @param l a std::initializer_list that contains the elements this List will store.
        */
        List(const std::initializer_list<T> &l) : Container<T>(l)
        {
            nElements = this->size;
        }

        /*
        Constructor with an Arrary as Input.
        @param arr a std::array that contains the elements this List will store.
        */
        template <std::size_t N>
        List(const std::array<T, N> &arr) : Container<T>(arr)
        {
            nElements = this->size;
        }

        /*
        Copy Constructor
        @param other a List to be copied.
        */
        List(const List<T> &other) : Container<T>(other)
        {
            nElements = this->nElements;
        }

        /*
        Move Constructor
        @param other a Vector to be moved.
        */
        List(List &&other) : Container<T>(other)
        {
            nElements = other.nElements;
        }

        /*
        Copy Assignment
        @param other a List.
        @return a reference to this List.
        */
        virtual List<T> &operator=(const List<T> &other)
        {
            Container<T>::operator=(other);
            nElements = this->nElements;
            return *this;
        }

        /*
        Operator []
        @param index the index of the element to be accessed.
        @return the element
        */
        virtual T &operator[](const std::size_t &index)
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
        Checks if this List is empty or not.
        @return a bool that indicates whether this List is empty.
        */
        virtual bool IsEmpty() const override { return nElements == 0; }

        /*
        Converts this List to a string that displays all the elements
        of this List.
        @return a string that represents this List.
        */
        virtual std::string ToString() const override
        {
            if (nElements == 0)
                return "[]";
            std::stringstream ss;
            ss << "[";
            for (std::size_t i = 0; i < nElements; i++)
            {
                ss << this->data[i];
                if (i < nElements - 1)
                    ss << ", ";
            }
            ss << "]";
            return ss.str();
        }

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

        /*
        Clears all the elements this List stores.
        */
        void Clear()
        {
            if (this->data) {
                delete[] this->data;
                this->data = nullptr;
            }
            nElements = 0;
            shrink();
        }
    };
} // namespace DataStructure

#endif