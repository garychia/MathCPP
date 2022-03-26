#ifndef CONTAINER_H
#define CONTAINER_H

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
        Container() : size(0), data(nullptr) {}

        /*
        Constructor with Initial Size and an Initial Value.
        @param s the initial size of the Container to be generated.
        @param value the value the Container will be filled with.
        */
        Container(std::size_t s, const T &value) : size(s)
        {
            data = new T[s];
            for (std::size_t i = 0; i < s; i++)
                data[i] = value;
        }

        /*
        Constructor with Initializer List as Input.
        @param l an initializer_list that contains the elements this Container will store.
        */
        Container(const std::initializer_list<T>& l) : size(l.size())
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
        Container(const std::array<T, N>& arr) : size(arr.size())
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
        Constructor with a std::vector.
        @param values a std::vector that contains the elements this Container
        will store.
        */
        Container(const std::vector<T> &values) : size(values.size())
        {
            if (size > 0)
            {
                data = new T[size];
                for (std::size_t i = 0; i < size; i++)
                {
                    data[i] = values[i];
                }
            }
            else
                data = nullptr;
        }

        /*
        Copy Constructor
        @param other a Container to be copied.
        */
        Container(const Container<T> &other)
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
        Container(Container<T> &&other)
        {
            size = move(other.size);
            data = move(other.data);
            other.size = 0;
            other.data = nullptr;
        }

        /*
        Destructor
        */
        virtual ~Container()
        {
            if (data)
                delete[] data;
        }

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
        virtual Container<T> &operator=(const Container<T> &other) {
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
        Returns the number of elements this Container stores.
        @return the number of elements this Container stores.
        */
        virtual std::size_t Size() const { return size; }

        /*
        Converts this container to a string that shows the elements
        of this Container.
        @return a string that represents this Container.
        */
        virtual std::string ToString() const = 0;

        /*
        Converts this Container to a string and pass it to an output stream.
        @param stream an output stream.
        @param t a Container
        @return a reference to the output stream.
        */
        friend std::ostream &operator<<(std::ostream &stream, const Container<T> &container)
        {
            stream << container.ToString();
            return stream;
        }
    };
} // namespace DataStructure

#endif