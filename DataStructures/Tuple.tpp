namespace DataStructure
{
    template <class T>
    Tuple<T>::Tuple() : Container<T>() {}

    template <class T>
    Tuple<T>::Tuple(std::size_t s, const T &value) : Container<T>(s, value) {}

    template <class T>
    Tuple<T>::Tuple(const std::initializer_list<T> &l) : Container<T>(l) {}

    template <class T>
    template <std::size_t N>
    Tuple<T>::Tuple(const std::array<T, N> &arr) : Container<T>(arr) {}

    template <class T>
    Tuple<T>::Tuple(const std::vector<T> &values) : Container<T>(values) {}

    template <class T>
    Tuple<T>::Tuple(const Container<T> &other) : Container<T>(other) {}

    template <class T>
    template <class OtherType>
    Tuple<T>::Tuple(const Container<OtherType> &other) : Container<T>(other) {}

    template <class T>
    Tuple<T>::Tuple(Container<T> &&other) : Container<T>(other) {}

    template <class T>
    template <class OtherType>
    Tuple<T>::Tuple(Container<OtherType> &&other) : Container<T>(other) {}

    template <class T>
    const T &Tuple<T>::operator[](const std::size_t &index) const
    {
        if (index < this->size)
            return this->data[index];
        else
            throw Exceptions::IndexOutOfBound(
                index,
                "Tuple: Index must be non-negative and less than the number of elements.");
    }

    template <class T>
    Tuple<T> &Tuple<T>::operator=(const Container<T> &other)
    {
        Container<T>::operator=(other);
        return *this;
    }

    template <class T>
    template <class OtherType>
    Tuple<T> &Tuple<T>::operator=(const Container<OtherType> &other)
    {
        Container<T>::operator=(other);
        return *this;
    }

    template <class T>
    template <class OtherType>
    bool Tuple<T>::operator==(const Tuple<OtherType> &other) const
    {
        if (this->Size() != other.Size())
            return false;
        for (std::size_t i = 0; i < this->Size(); i++)
            if ((*this)[i] != other[i])
                return false;
        return true;
    }

    template <class T>
    template <class OtherType>
    bool Tuple<T>::operator!=(const Tuple<OtherType> &other) const
    {
        return !operator==(other);
    }

    template <class T>
    std::string Tuple<T>::ToString() const
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

}