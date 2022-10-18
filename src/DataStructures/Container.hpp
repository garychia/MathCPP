#ifndef CONTAINER_HPP
#define CONTAINER_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <vector>

namespace DataStructures {
/*
Container is an abstract class that is capable of storing data.
*/
template <class T> class Container {
protected:
  // number of elements stored
  size_t size;
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
  Container(size_t s, const T &value) : size(s), data(nullptr) {
    if (!s)
      return;
    data = new T[s];
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < s; i++)
      data[i] = value;
  }

  /*
  Constructor with Initializer List as Input.
  @param l an initializer_list that contains the elements this Container
  will store.
  */
  Container(const std::initializer_list<T> &l) : size(l.size()), data(nullptr) {
    if (!size)
      return;
    data = new T[size];
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; i++)
      data[i] = *(l.begin() + i);
  }

  /*
  Constructor with arrary as Input.
  @param arr an array that contains the elements this Container will store.
  */
  template <size_t N>
  Container(const std::array<T, N> &arr) : size(arr.size()), data(nullptr) {
    if (!size)
      return;
    data = new T[size];
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; i++)
      data[i] = arr[i];
  }

  /*
  Constructor with a std::vector.
  @param values a std::vector that contains the elements this Container
  will store.
  */
  Container(const std::vector<T> &values) : size(values.size()), data(nullptr) {
    if (size) {
      data = new T[size];
#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < size; i++)
        data[i] = values[i];
    }
  }

  /*
  Copy Constructor
  @param other a Container to be copied.
  */
  Container(const Container<T> &other) : Container(other.Size(), 0) {
    if (!size)
      return;
    data = new T[size];
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; i++)
      data[i] = other[i];
  }

  /*
  Copy Constructor
  @param other a Container to be copied.
  */
  template <class OtherType>
  Container(const Container<OtherType> &other) : Container(other.Size(), 0) {
    if (!size)
      return;
    data = new T[size];
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < size; i++)
      data[i] = (T)other[i];
  }

  /*
  Move Constructor
  @param other a Container to be moved.
  */
  Container(Container<T> &&other) : size(other.Size()), data(other.data) {
    other.size = 0;
    other.data = nullptr;
  }

  /*
  Destructor
  */
  virtual ~Container() {
    if (data)
      delete[] data;
  }

  /*
  Access the element of a given index.
  @param index the index of the element to be accessed.
  @return the reference of the element accessed.
  */
  virtual const T &operator[](const size_t &index) const = 0;

  /*
  Copy Assignment
  @param other a Container to be copied.
  @return a reference to this Container.
  */
  virtual Container<T> &operator=(const Container<T> &other) {
    if (&other == this)
      return *this;
    size = other.Size();
    if (data)
      delete[] data;
    data = nullptr;
    if (size) {
      data = new T[size];
#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < size; i++)
        data[i] = T(other[i]);
    }
    return *this;
  }

  /*
  Copy Assignment
  @param other a Container containing values of a different type to be copied.
  @return a reference to this Container.
  */
  template <class OtherType>
  Container<T> &operator=(const Container<OtherType> &other) {
    size = other.Size();
    if (data)
      delete[] data;
    data = nullptr;
    if (size > 0) {
      data = new T[size];
#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < size; i++)
        data[i] = T(other[i]);
    }
    return *this;
  }

  /*
  Returns the number of elements this Container stores.
  @return the number of elements this Container stores.
  */
  virtual size_t Size() const { return size; }

  /*
  Checks if this Container is empty or not.
  @return a bool that indicates whether this Container is empty.
  */
  virtual bool IsEmpty() const { return !size; }

  /*
  Converts this container to a string that shows the elements
  of this Container.
  @return a string that represents this Container.
  */
  virtual std::string ToString() const = 0;

  friend std::ostream;

  template <class OtherType> friend class Container;
};

/**
* Converts this Container to a string and pass it to an output stream.
* @param stream an output stream.
* @param t a Container
* @return a reference to the output stream.
*/
template <class T>
std::ostream &operator<<(std::ostream &stream, const Container<T> &container) {
  stream << container.ToString();
  return stream;
}

} // namespace DataStructures

#endif
