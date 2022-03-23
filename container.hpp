#ifndef CONTAINER_H
#define CONTAINER_H

template <class T>
class Container
{
public:
    virtual T &operator[](const std::size_t &index) = 0;
    virtual std::size_t Size() const = 0;
};

#endif