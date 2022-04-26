namespace DataStructure
{
    template <class T>
    T &Vector3D<T>::X() { return this->data[X_INDEX]; }
    
    template <class T>
    T &Vector3D<T>::Y() { return this->data[Y_INDEX]; }
    
    template <class T>
    T &Vector3D<T>::Z() { return this->data[Z_INDEX]; }

    template <class T>
    Vector3D<T>::Vector3D(T x, T y, T z) : Container<T>(NUM_COMPONENTS, 0)
    {
        this->data[X_INDEX] = x;
        this->data[Y_INDEX] = y;
        this->data[Z_INDEX] = z;
    }

    template <class T>
    Vector3D<T>::Vector3D(const Vector3D<T> &other) : Container<T>(other) {}

    template <class T>
    template <class OtherType>
    Vector3D<T>::Vector3D(const Vector3D<OtherType> &other) : Container<T>(other) {}

    template <class T>
    Vector3D<T>::Vector3D(Vector3D<T> &&other) : Container<T>(other) {}

    template <class T>
    template <class OtherType>
    Vector3D<T>::Vector3D(Vector3D<T> &&other) : Container<T>(other) {}

    template <class T>
    Vector3D<T> &Vector3D<T>::operator=(const Vector3D<T> &other)
    {
        Container<T>::operator=(other);
        return *this;
    }

    template <class T>
    std::string Vector3D<T>::ToString() const
    {
        std::stringstream ss;
        ss << "(" << this->data[X_INDEX] << ", "
           << this->data[Y_INDEX] << ", "
           << this->data[Z_INDEX] << ")";
        return ss.str();
    }

    template <class T>
    T &Vector3D<T>::operator[](const std::size_t &index)
    {
        switch (index)
        {
        case X_INDEX:
            return this->data[X_INDEX];
        case Y_INDEX:
            return this->data[Y_INDEX];
        case Z_INDEX:
            return this->data[Z_INDEX];
        default:
            throw Exceptions::IndexOutOfBound(
                index,
                "Vector3D: Index must be between 0 and 2 inclusively.");
        }
    }

    template <class T>
    const T &Vector3D<T>::operator[](const std::size_t &index) const
    {
        switch (index)
        {
        case X_INDEX:
            return this->data[X_INDEX];
        case Y_INDEX:
            return this->data[Y_INDEX];
        case Z_INDEX:
            return this->data[Z_INDEX];
        default:
            throw Exceptions::IndexOutOfBound(
                index,
                "Vector3D: Index must be between 0 and 2 inclusively.");
        }
    }

    template <class T>
    std::size_t Vector3D<T>::Size() const { return NUM_COMPONENTS; }

    template <class T>
    template <class ReturnType>
    ReturnType Vector3D<T>::Length() const
    {
        return std::sqrt(
            this->data[X_INDEX] * this->data[X_INDEX] +
            this->data[Y_INDEX] * this->data[Y_INDEX] +
            this->data[Z_INDEX] * this->data[Z_INDEX]);
    }

    template <class T>
    template <class OtherType>
    auto Vector3D<T>::Add(const Vector3D<OtherType> &other) const
    {
        return Vector3D<decltype(this->data[X_INDEX] + other[X_INDEX])>(
            this->data[X_INDEX] + other[X_INDEX],
            this->data[Y_INDEX] + other[Y_INDEX],
            this->data[Z_INDEX] + other[Z_INDEX]);
    }

    template <class T>
    template <class OtherType>
    auto Vector3D<T>::operator+(const Vector3D<OtherType> &other) const
    {
        return this->Add(other);
    }

    template <class T>
    template <class OtherType>
    Vector3D<T> &Vector3D<T>::operator+=(const Vector3D<OtherType> &other)
    {
        this->data[X_INDEX] += other[X_INDEX];
        this->data[Y_INDEX] += other[Y_INDEX];
        this->data[Z_INDEX] += other[Z_INDEX];
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector3D<T>::Minus(const Vector3D<OtherType> &other) const
    {
        return Vector3D<decltype(this->data[X_INDEX] - other[X_INDEX])>(
            this->data[X_INDEX] - other[X_INDEX],
            this->data[Y_INDEX] - other[Y_INDEX],
            this->data[Z_INDEX] - other[Z_INDEX]);
    }

    template <class T>
    template <class OtherType>
    auto Vector3D<T>::operator-(const Vector3D<OtherType> &other) const
    {
        return this->Minus(other);
    }

    template <class T>
    template <class OtherType>
    Vector3D<T> &Vector3D<T>::operator-=(const Vector3D<OtherType> &other)
    {
        this->data[X_INDEX] -= other[X_INDEX];
        this->data[Y_INDEX] -= other[Y_INDEX];
        this->data[Z_INDEX] -= other[Z_INDEX];
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector3D<T>::Scale(const OtherType &scaler) const
    {
        return Vector3D<decltype(this->data[X_INDEX] * scaler)>(
            this->data[X_INDEX] * scaler,
            this->data[Y_INDEX] * scaler,
            this->data[Z_INDEX] * scaler);
    }

    template <class T>
    template <class OtherType>
    auto Vector3D<T>::operator*(const OtherType &scaler) const
    {
        return this->Scale(scaler);
    }

    template <class T>
    template <class OtherType>
    Vector3D<T> &Vector3D<T>::operator*=(const OtherType &scaler)
    {
        this->data[X_INDEX] *= scaler;
        this->data[Y_INDEX] *= scaler;
        this->data[Z_INDEX] *= scaler;
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector3D<T>::Divide(const OtherType &scaler) const
    {
        return Vector3D<decltype(this->data[X_INDEX] / scaler)>(
            this->data[X_INDEX] / scaler,
            this->data[Y_INDEX] / scaler,
            this->data[Z_INDEX] / scaler);
    }

    template <class T>
    template <class OtherType>
    auto Vector3D<T>::operator/(const OtherType &scaler) const
    {
        return this->Divide(scaler);
    }

    template <class T>
    template <class OtherType>
    Vector3D<T> &Vector3D<T>::operator/=(const OtherType &scaler)
    {
        this->data[X_INDEX] /= scaler;
        this->data[Y_INDEX] /= scaler;
        this->data[Z_INDEX] /= scaler;
        return *this;
    }

    template <class T>
    template <class OtherType>
    decltype(auto) Vector3D<T>::Dot(const Vector3D<OtherType> &other) const
    {
        return this->data[X_INDEX] * other[X_INDEX] +
               this->data[Y_INDEX] * other[Y_INDEX] +
               this->data[Z_INDEX] * other[Z_INDEX];
    }

    template <class T>
    template <class OtherType>
    auto Vector3D<T>::Cross(const Vector3D<OtherType> &other) const
    {
        return Vector3D<decltype(this->data[Y_INDEX] * other[Z_INDEX])>(
            this->data[Y_INDEX] * other[Z_INDEX] - other[Y_INDEX] * this->data[Z_INDEX],
            this->data[Z_INDEX] * other[X_INDEX] - other[Z_INDEX] * this->data[X_INDEX],
            this->data[X_INDEX] * other[Y_INDEX] - other[X_INDEX] * this->data[Y_INDEX]);
    }

    template <class T>
    Vector3D<T> Vector3D<T>::Normalized() const
    {
        const T length = Length();
        if (length == 0)
            throw Exceptions::DividedByZero("Cannot normalize a zero vector.");
        return *this / length;
    }

    template <class T>
    void Vector3D<T>::Normalize()
    {
        const T length = Length();
        if (length == 0)
            throw Exceptions::DividedByZero("Vector3D: Cannot normalize a zero vector.");
        *this /= length;
    }

    template <class T>
    std::ostream &operator<<(std::ostream &stream, const Vector3D<T> &v)
    {
        stream << v.ToString();
        return stream;
    }

}