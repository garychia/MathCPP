namespace DataStructure
{
    template <class T>
    Container<T>::Container() : size(0), data(nullptr) {}

    template <class T>
    Container<T>::Container(std::size_t s, const T &value) : size(s)
    {
        if (s > 0)
        {
            data = new T[s];
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < s; i++)
                data[i] = value;
        }
        else
            data = nullptr;
    }

    template <class T>
    Container<T>::Container(const std::initializer_list<T> &l) : size(l.size())
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

    template <class T>
    template <std::size_t N>
    Container<T>::Container(const std::array<T, N> &arr) : size(arr.size())
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

    template <class T>
    Container<T>::Container(const std::vector<T> &values) : size(values.size())
    {
        if (size > 0)
        {
            data = new T[size];
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < size; i++)
                data[i] = values[i];
        }
        else
            data = nullptr;
    }

    template <class T>
    Container<T>::Container(const Container<T> &other)
    {
        size = other.Size();
        if (size > 0)
        {
            T *newData = new T[size];
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < size; i++)
                newData[i] = (T)other[i];
            data = newData;
        }
        else
            data = nullptr;
    }

    template <class T>
    template <class OtherType>
    Container<T>::Container(const Container<OtherType> &other)
    {
        size = other.Size();
        if (size > 0)
        {
            T *newData = new T[size];
#pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < size; i++)
                newData[i] = (T)other[i];
            data = newData;
        }
        else
            data = nullptr;
    }

    template <class T>
    Container<T>::Container(Container<T> &&other)
    {
        size = std::move(other.size);
        data = std::move(other.data);
        other.size = 0;
        other.data = nullptr;
    }

    template <class T>
    template <class OtherType>
    Container<T>::Container(Container<OtherType> &&other)
    {
        size = move(other.size);
        data = move(other.data);
        other.size = 0;
        other.data = nullptr;
    }

    template <class T>
    Container<T>::~Container()
    {
        if (data)
            delete[] data;
    }

    template <class T>
    Container<T> &Container<T>::operator=(const Container<T> &other)
    {
        if (this != &other)
        {
            size = other.size;
            if (data)
                delete[] data;
            data = nullptr;
            if (size > 0)
            {
                data = new T[size];
#pragma omp parallel for schedule(dynamic)
                for (std::size_t i = 0; i < size; i++)
                    data[i] = other.data[i];
            }
        }
        return *this;
    }

    template <class T>
    template <class OtherType>
    Container<T> &Container<T>::operator=(const Container<OtherType> &other)
    {
        if (this != &other)
        {
            size = other.size;
            if (data)
                delete[] data;
            data = nullptr;
            if (size > 0)
            {
                data = new T[size];
#pragma omp parallel for schedule(dynamic)
                for (std::size_t i = 0; i < size; i++)
                    data[i] = (T)other.data[i];
            }
        }
        return *this;
    }

    template <class T>
    std::size_t Container<T>::Size() const { return size; }

    template <class T>
    bool Container<T>::IsEmpty() const { return size == 0; }

    /*
    Converts this Container to a string and pass it to an output stream.
    @param stream an output stream.
    @param t a Container
    @return a reference to the output stream.
    */
    template <class T>
    std::ostream &operator<<(std::ostream &stream, const Container<T> &container)
    {
        stream << container.ToString();
        return stream;
    }
}