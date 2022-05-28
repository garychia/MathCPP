namespace DataStructure
{
    template <class T>
    Vector<T>::Vector() : Tuple<T>() {}

    template <class T>
    Vector<T>::Vector(std::size_t s, const T &value) : Tuple<T>(s, value) {}

    template <class T>
    Vector<T>::Vector(const std::initializer_list<T> &l) : Tuple<T>(l) {}

    template <class T>
    template <std::size_t N>
    Vector<T>::Vector(const std::array<T, N> &arr) : Tuple<T>(arr) {}

    template <class T>
    Vector<T>::Vector(const Container<T> &other) : Tuple<T>(other) {}

    template <class T>
    template <class OtherType>
    Vector<T>::Vector(const Container<OtherType> &other) : Tuple<T>(other) {}

    template <class T>
    Vector<T>::Vector(Container<T> &&other) : Tuple<T>(other) {}

    template <class T>
    template <class OtherType>
    Vector<T>::Vector(Container<OtherType> &&other) : Tuple<T>(other) {}

    template <class T>
    Vector<T> &Vector<T>::operator=(const Container<T> &other)
    {
        Tuple<T>::operator=(other);
        return *this;
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator=(const Container<OtherType> &other)
    {
        Tuple<T>::operator=(other);
        return *this;
    }

    template <class T>
    T &Vector<T>::operator[](const std::size_t &index)
    {
        if (index > this->size - 1)
            throw Exceptions::IndexOutOfBound(
                index,
                "Vector: Index must be less than the dimention.");
        return this->data[index];
    }

    template <class T>
    const T &Vector<T>::operator[](const std::size_t &index) const
    {
        if (index > this->size - 1)
            throw Exceptions::IndexOutOfBound(
                index,
                "Vector: Index must be less than the dimention.");
        return this->data[index];
    }

    template <class T>
    std::size_t Vector<T>::Dimension() const { return this->size; }

    template <class T>
    template <class ReturnType>
    ReturnType Vector<T>::Length() const
    {
        return LpNorm((ReturnType)2);
    }

    template <class T>
    template <class ReturnType>
    ReturnType Vector<T>::LpNorm(ReturnType p) const
    {
        ReturnType squaredTotal = 0;
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < this->size; i++)
        {
            ReturnType squaredElement = std::pow(this->data[i], p);
#pragma omp atomic
            squaredTotal += squaredElement;
        }
        return std::pow(squaredTotal, 1 / p);
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Add(const Vector<OtherType> &other) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform addition on an empty vector.");
        else if (other.size == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform addtion on the given empty vector.");
        else if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expected the dimension of the second vector to be a factor of that of the first.");
        Vector<decltype(this->data[0] + other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] += other[i % other.Dimension()];
        return result;
    }

    template <class T>
    template <class ScalerType>
    auto Vector<T>::Add(const ScalerType &scaler) const
    {
        Vector<decltype(this->data[0] + scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] += scaler;
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator+(const Vector<OtherType> &other) const
    {
        try
        {
            return this->Add(other);
        }
        catch (Exceptions::EmptyVector &e)
        {
            throw e;
        }
        catch (Exceptions::InvalidArgument &e)
        {
            throw e;
        }
    }

    template <class T>
    template <class ScalerType>
    auto Vector<T>::operator+(const ScalerType &scaler) const
    {
        return this->Add(scaler);
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator+=(const Vector<OtherType> &other)
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform addition on an empty vector.");
        else if (other.Size() == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform addtion on the given empty vector.");
        else if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expected the dimension of the second vector to be a factor of that of the first.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            this->data[i] += other[i % other.Dimension()];
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Minus(const Vector<OtherType> &other) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform subtraction on an empty vector.");
        else if (other.size == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform subtraction on the given empty vector.");
        else if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expected the dimension of the second vector to be a factor of that of the first.");

        Vector<decltype(this->data[0] - other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] -= other[i % other.Dimension()];
        return result;
    }

    template <class T>
    template <class ScalerType>
    auto Vector<T>::Minus(const ScalerType &scaler) const
    {
        Vector<decltype(this->data[0] - scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] -= scaler;
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator-(const Vector<OtherType> &other) const
    {
        try
        {
            return this->Minus(other);
        }
        catch (Exceptions::EmptyVector &e)
        {
            throw e;
        }
        catch (Exceptions::InvalidArgument &e)
        {
            throw e;
        }
    }

    template <class T>
    template <class ScalerType>
    auto Vector<T>::operator-(const ScalerType &scaler) const
    {
        return this->Minus(scaler);
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator-=(const Vector<OtherType> &other)
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform subtraction on an empty vector.");
        else if (other.Size() == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform subtraction on the given empty vector.");
        else if (Dimension() % other.Dimension() != 0)
            throw Exceptions::InvalidArgument(
                "Vector: Expected the dimension of the second vector to be a factor of that of the first.");

#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            this->data[i] -= other[i % other.Dimension()];
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Scale(const OtherType &scaler) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform scaling on an empty vector.");
        Vector<decltype(this->data[0] * scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] *= scaler;
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator*(const OtherType &scaler) const
    {
        try
        {
            return this->Scale(scaler);
        }
        catch (Exceptions::EmptyVector &e)
        {
            throw e;
        }
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator*(const Vector<OtherType> &other) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform scaling on an empty vector.");
        if (this->Dimension() != other.Dimension())
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform elmenet-wise multiplication on vectors with different dimensions.");
        Vector<decltype((*this)[0] * other[0])> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < this->size; i++)
            result[i] *= other[i];
        return result;
    }

    template <class T>
    Vector<T> &Vector<T>::operator*=(const T &scaler)
    {
        static_assert(!std::is_base_of<Vector, T>::value, "OtherType must be a scaler.");
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform scaling on an empty vector.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            this->data[i] *= scaler;
        return *this;
    }

    template <class T>
    Vector<T> &Vector<T>::operator*=(const Vector<T> &other)
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform element-wise multiplication on an empty vector.");
        if (Dimension() != other.Dimension())
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform element-wise multiplication on vectors of "
                "different dimensions.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            this->data[i] *= other[i];
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Divide(const OtherType &scaler) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform division on an empty vector.");
        else if (scaler == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot divide a vector by 0.");
        Vector<decltype(this->data[0] / scaler)> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            result[i] /= scaler;
        return result;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::operator/(const OtherType &scaler) const
    {
        try
        {
            return this->Divide(scaler);
        }
        catch (Exceptions::EmptyVector &e)
        {
            throw e;
        }
        catch (Exceptions::InvalidArgument &e)
        {
            throw e;
        }
    }

    template <class T>
    template <class OtherType>
    Vector<T> &Vector<T>::operator/=(const OtherType &scaler)
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform division on an empty vector.");
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
            this->data[i] /= scaler;
        return *this;
    }

    template <class T>
    template <class OtherType>
    auto Vector<T>::Dot(const Vector<OtherType> &other) const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform subtraction on an empty vector.");
        else if (other.size == 0)
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform subtraction on the given empty vector.");
        else if (Dimension() != other.Dimension())
            throw Exceptions::InvalidArgument(
                "Vector: Cannot perform subtraction on vectors with different dimensions.");
        decltype(this->data[0] * other[0]) result = 0;
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < Dimension(); i++)
        {
            decltype(result) mult = this->data[i] * other[i];
#pragma omp atomic
            result += mult;
        }
        return result;
    }

    template <class T>
    Vector<T> Vector<T>::Normalized() const
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform normalization on an empty vector.");
        const T length = Length<T>();
        if (length == 0)
            throw Exceptions::DividedByZero("Vector: Cannot normalize a zero vector.");
        return *this / length;
    }

    template <class T>
    void Vector<T>::Normalize()
    {
        if (this->size == 0)
            throw Exceptions::EmptyVector(
                "Vector: Cannot perform normalization on an empty vector.");
        const T length = Length<T>();
        if (length == 0)
            throw Exceptions::DividedByZero("Vector: Cannot normalize a zero vector.");
        *this /= length;
    }

    template <class T>
    T Vector<T>::Sum() const
    {
        T total = 0;
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < this->size; i++)
            total += (*this)[i];
        return total;
    }

    template <class T>
    template <class OtherType>
    Vector<OtherType> Vector<T>::Map(const std::function<OtherType(T)> &f) const
    {
        Vector<OtherType> result(*this);
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < this->size; i++)
            result[i] = f(result[i]);
        return result;
    }

    template <class T>
    Vector<T> Vector<T>::ZeroVector(const std::size_t &n)
    {
        Vector<T> v;
        v.size = n;
        v.data = new T[n];
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < n; i++)
            v.data[i] = 0;
        return v;
    }

    template <class T>
    Vector<T> Vector<T>::Combine(const std::initializer_list<Vector<T>> &vectors)
    {
        std::size_t elementTotal = 0;
        for (auto itr = vectors.begin(); itr != vectors.end(); itr++)
        {
            elementTotal += itr->Size();
        }
        Vector<T> combined(elementTotal, 0);
        std::size_t currentIndex = 0;
        for (auto vector : vectors)
            for (std::size_t j = 0; j < vector.Size(); j++)
                combined[currentIndex++] = vector[j];
        return combined;
    }
}