namespace MLAlgs
{
    /*
    Sign function.
    @param value a value.
    @return +1 if the value is positive, or 0 if the value is 0. -1, otherwise.
    */
    template <class T>
    T Sign(T value)
    {
        return value == 0 ? 0 : (value > 0 ? 1 : -1);
    }
} // namespace MLAlgs
