namespace DataStructure
{
    template <class T>
    void List<T>::resize()
    {
        this->size = this->size > 2 ? this->size << 1 : INITIAL_SIZE;
        T *newData = new T[this->size];
        for (std::size_t i = 0; i < nElements; i++)
            newData[i] = this->data[i];
        if (this->data)
            delete[] this->data;
        this->data = newData;
    }

    template <class T>
    void List<T>::shrink()
    {
        this->size = this->size / 2 > INITIAL_SIZE ? this->size / 2 : INITIAL_SIZE;
        T *newData = new T[this->size];
        for (std::size_t i = 0; i < nElements; i++)
            newData[i] = this->data[i];
        if (this->data)
            delete[] this->data;
        this->data = newData;
    }

    template <class T>
    List<T>::List() : Container<T>(), nElements(0)
    {
        this->data = new T[INITIAL_SIZE];
        this->size = INITIAL_SIZE;
    }

    template <class T>
    List<T>::List(std::size_t s, const T &value) : Container<T>(s, value), nElements(s) {}

    template <class T>
    List<T>::List(const std::initializer_list<T> &l) : Container<T>(l)
    {
        nElements = this->size;
    }

    template <class T>
    template <std::size_t N>
    List<T>::List(const std::array<T, N> &arr) : Container<T>(arr)
    {
        nElements = this->size;
    }

    template <class T>
    List<T>::List(const Container<T> &other) : Container<T>(other)
    {
        nElements = other.Size();
    }

    template <class T>
    List<T>::List(Container<T> &&other) : Container<T>(other)
    {
        nElements = other.size;
    }

    template <class T>
    List<T> &List<T>::operator=(const Container<T> &other)
    {
        Container<T>::operator=(other);
        nElements = other.Size();
        return *this;
    }

    template <class T>
    T &List<T>::operator[](const std::size_t &index)
    {
        if (index > nElements - 1)
            throw Exceptions::IndexOutOfBound(
                index,
                "Vector: Index must be less than the dimention.");
        return this->data[index];
    }

    template <class T>
    const T & List<T>::operator[](const std::size_t &index) const
    {
        if (index > nElements - 1)
            throw Exceptions::IndexOutOfBound(
                index,
                "List: Index must be less than the number of elements.");
        return this->data[index];
    }

    template <class T>
    std::size_t List<T>::Size() const { return nElements; }

    template <class T>
    bool List<T>::IsEmpty() const { return nElements == 0; }

    template <class T>
    std::string List<T>::ToString() const
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

    template <class T>
    void List<T>::Append(const T &element)
    {
        if (nElements + 1 > this->size)
            resize();
        this->data[nElements++] = element;
    }

    template <class T>
    void List<T>::Append(T &&element)
    {
        if (nElements + 1 > this->size)
            resize();
        this->data[nElements++] = std::move(element);
    }

    template <class T>
    void List<T>::Prepend(const T &element)
    {
        if (nElements + 1 > this->size)
            resize();
        nElements++;
        for (std::size_t i = nElements - 1; i > 0; i--)
            this->data[i] = this->data[i - 1];
        this->data[0] = element;
    }

    template <class T>
    void List<T>::Prepend(T &&element)
    {
        if (nElements + 1 > this->size)
            resize();
        nElements++;
        for (std::size_t i = nElements - 1; i > 0; i--)
            this->data[i] = this->data[i - 1];
        this->data[0] = std::move(element);
    }

    template <class T>
    T List<T>::PopEnd()
    {
        if (nElements == 0)
            throw Exceptions::EmptyList("List: Trying to pop an element from an empty list.");
        T element = this->data[--nElements];
        if (nElements < this->size / 2)
            shrink();
        return element;
    }

    template <class T>
    T List<T>::PopFront()
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

    template <class T>
    void List<T>::Clear()
    {
        if (this->data)
        {
            delete[] this->data;
            this->data = nullptr;
        }
        nElements = 0;
        shrink();
    }
} // namespace DataStructure