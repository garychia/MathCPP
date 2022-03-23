#ifndef CONTAINER_H
#define CONTAINER_H

/*
Container is an abstract class that is capable of storing data.
*/
template <class T>
class Container
{
public:
    /*
    Access the element of a given index.
    @param index the index of the element to be accessed.
    @return the reference of the element accessed.
    */
    virtual T &operator[](const std::size_t &index) = 0;

    /*
    Returns the number of elements this Container stores.
    @return the number of elements this Container stores.
    */
    virtual std::size_t Size() const = 0;
};

#endif