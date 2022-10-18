#ifndef TUPLE_HPP
#define TUPLE_HPP

#include <vector>
#include <sstream>

#include "Container.hpp"

namespace DataStructures {
/* An immutable container of a fixed size. */
template <class T> class Tuple : public Container<T> {
public:
  /* Constructor that Generates an Empty Tuple. */
  Tuple() : Container<T>() {}

  /**
   * Constructor that generates a Tuple filled with identical values.
   * @param s the size (number of elements) of the Tuple.
   * @param value the value the Tuple will be filled with.
   **/
  Tuple(size_t s, const T &value) : Container<T>(s, value) {}

  /**
   * Constructor with a std::initializer_list as Input.
   * @param l a std::initializer_list that contains the elements this Tuple will
   *store.
   **/
  Tuple(const std::initializer_list<T> &l) : Container<T>(l) {}

  /**
   * Constructor with a std::arrary as Input.
   * @param arr a std::array that contains the elements this Tuple will store.
   **/
  template <size_t N>
  Tuple(const std::array<T, N> &arr) : Container<T>(arr) {}

  /**
   * Constructor with a std::vector as Input.
   * @param values a std::vector that contains the elements this Tuple will
   *store.
   **/
  Tuple(const std::vector<T> &values) : Container<T>(values) {}

  /**
   * Copy Constructor.
   * @param other a Container whose elements the Tuple will copy.
   **/
  Tuple(const Container<T> &other) : Container<T>(other) {}

  /**
   * Copy Constructor with a Container whose elements are of a different type.
   * @param other a Container whose elements the Tuple will copy.
   **/
  template <class OtherType>
  Tuple(const Container<OtherType> &other) : Container<T>(other) {}

  /**
   * Move Constructor.
   * @param other a Container whose elements will be 'moved' into the Tuple.
   **/
  Tuple(Container<T> &&other) : Container<T>(std::move(other)) {}

  /**
   * Access the element at a given index.
   * @param index the index at which the element will be accessed.
   * @return the reference to the element accessed.
   * @throw IndexOutOfBound when the index exceeds the largest possible index.
   **/
  virtual const T &operator[](const size_t &index) const override {
    if (index < this->size)
      return this->data[index];
    throw Exceptions::IndexOutOfBound(index,
                                      "Tuple: Index must be non-negative and "
                                      "less than the number of elements.");
  }

  /**
   * Copy Assignment.
   * @param other a Container whose elements will be copied into the Tuple.
   * @return a reference to the Tuple.
   **/
  virtual Tuple<T> &operator=(const Container<T> &other) override {
    Container<T>::operator=(other);
    return *this;
  }

  /**
   * Copy Assignment
   * @param other a Container containing elements of a different type to be
   *copied.
   * @return a reference to the Tuple.
   **/
  template <class OtherType>
  Tuple<T> &operator=(const Container<OtherType> &other) {
    Container<T>::operator=(other);
    return *this;
  }

  virtual Tuple<T> &operator=(Container<T> &&other) noexcept {
    Container<T>::operator=(std::move(other));
    return *this;
  }

  /**
   * Check if two Tuples have the same elements.
   * @param other a Tuple to be compared with this Tuple.
   * @return a bool that indicates whether both Tuples have the same elements.
   **/
  template <class OtherType>
  bool operator==(const Tuple<OtherType> &other) const {
    // Check if the same Tuple is being compared.
    if (this == &other)
      return true;
    // Both must have the same size.
    if (this->Size() != other.Size())
      return false;
    // Check if each pair of elements have identical values.
    for (std::size_t i = 0; i < this->Size(); i++)
      if ((*this)[i] != other[i])
        return false;
    return true;
  }

  /**
   * Check if two Tuples have different elements.
   * @param other a Tuple to be compared with this Tuple.
   * @return bool that indicates whether both Tuples have different elements.
   **/
  template <class OtherType>
  bool operator!=(const Tuple<OtherType> &other) const {
    return !operator==(other);
  }

  /**
   * Generate a string that displays the elements of the Tuple.
   * @return a string that displays the elements of the Tuple.
   **/
  virtual std::string ToString() const override {
    std::stringstream ss;
    ss << "(";
    for (std::size_t i = 0; i < this->size; i++) {
      ss << this->data[i];
      if (i < this->size - 1)
        ss << ", ";
    }
    ss << ")";
    return ss.str();
  }

  template <class OtherType> friend class Tuple;
};
} // namespace DataStructures

#endif
