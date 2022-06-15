#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <initializer_list>
#include <array>
#include <vector>

#include "Tuple.hpp"

namespace DataStructures
{
    /* A mutable Container of a fixed size that supports numerical operations. */
    template <class T>
    class Vector : public Tuple<T>
    {
    public:
        /* Constructor that Constructs an Empty Vector. */
        Vector();

        /**
         * Constructor with Initial Size and a Value.
         * @param s the initial size of the Vector to be generated.
         * @param value the value the Vector will be filled with.
         **/
        Vector(std::size_t s, const T &value);

        /**
         * Constructor with a std::initializer_list as Input.
         * @param l a std::initializer_list that contains the elements the Vector will store.
         **/
        Vector(const std::initializer_list<T> &l);

        /**
         * Constructor with a std::arrary as Input.
         * @param arr a std::array that contains the elements the Vector will store.
         **/
        template <std::size_t N>
        Vector(const std::array<T, N> &arr);

        /**
         * Constructor with a std::vector as Input.
         * @param v a std::vector that contains the elements the Vector will store.
         **/
        Vector(const std::vector<T> &v);

        /**
         * Copy Constructor.
         * @param other a Container whose elements will be copied into the Vector.
         **/
        Vector(const Container<T> &other);

        /**
         * Copy Constructor.
         * @param other a Container to be copied.
         **/
        template <class OtherType>
        Vector(const Container<OtherType> &other);

        /**
         * Move Constructor.
         * @param other a Container whose elements will be 'moved' into the Vector.
         **/
        Vector(Container<T> &&other);

        /**
         * Copy Assignment.
         * @param other a Container whose elements will be copied into this Vector.
         * @return a reference to this Vector.
         **/
        virtual Vector<T> &operator=(const Container<T> &other) override;

        /**
         * Copy Assignment.
         * @param other a Container that contains values of a different type.
         * @return a reference to this Vector.
         **/
        template <class OtherType>
        Vector<T> &operator=(const Container<OtherType> &other);

        /**
         * Access the element at a given index.
         * @param index the index at which the element will be accessed.
         * @return a reference to the accessed element.
         * @throw IndexOutOfBound when the index exceeds the largest possible index.
         **/
        virtual T &operator[](const std::size_t &index);

        /**
         * Access the element at a given index.
         * @param index the index at which the element will be accessed.
         * @return a reference to the accessed element.
         * @throw IndexOutOfBound when the index exceeds the largest possible index.
         **/
        virtual const T &operator[](const std::size_t &index) const override;

        /**
         * Return the dimention (number of values) of the Vector.
         * @return the dimention.
         **/
        std::size_t Dimension() const;

        /**
         * Return the Euclidean norm of the Vector.
         * @return the Euclidean norm.
         * @throw EmptyVector if the Vector is empty.
         **/
        template <class ReturnType>
        ReturnType Length() const;

        /**
         * Return the Euclidean norm of the Vector. (Same as Vector::Length)
         * @return the Euclidean norm.
         * @throw EmptyVector if the Vector is empty.
         **/
        template <class ReturnType>
        ReturnType EuclideanNorm() const;

        /**
         * Returns the Lp Norm of the Vector.
         * @return the Lp norm.
         * @throw EmptyVector if the Vector is empty.
         **/
        template<class ReturnType>
        ReturnType LpNorm(int p) const;

        /**
         * Perform addition with two Vectors.
         * @param other a Vector as the second operand.
         * @return a new Vector that is the sum of the Vectors.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty or the number of elements of
         * the second operand is not a factor of that of this Vector.
         **/
        template <class OtherType>
        auto Add(const Vector<OtherType> &other) const;

        /**
         * Perform element-wise addition with a Vector and a scaler.
         * @param scaler a scaler to be added to each element of the Vector.
         * @return a new Vector as the addition result.
         * @throw EmptyVector if the Vector is empty.
         **/
        template <class ScalerType>
        auto Add(const ScalerType &scaler) const;

        /**
         * Perform addition with two Vectors. (See Vector::Add)
         * @param other a Vector to be added to this Vector.
         * @return a new Vector as the addition result.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty or the number of elements of
         * the second operand is not a factor of that of this Vector.
         **/
        template <class OtherType>
        auto operator+(const Vector<OtherType> &other) const;

        /**
         * Perform element-wise addition with a Vector and a scaler. (See Vector::Add)
         * @param scaler a scaler to be added to each element of the Vector.
         * @return a new Vector as the addition result.
         * @throw EmptyVector if the Vector is empty.
         **/
        template <class ScalerType>
        auto operator+(const ScalerType &scaler) const;

        /**
         * Perform inplace addition with another Vector.
         * @param other a Vector to be added to this Vector.
         * @return the reference to this Vector.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty or the number of elements of
         * the second operand is not a factor of that of this Vector.
         **/
        template <class OtherType>
        Vector<T> &operator+=(const Vector<OtherType> &other);

        /**
         * Perform inplace addition with a scaler.
         * @param scaler a scaler to be added to each element of the Vector.
         * @return the reference to this Vector.
         * @throw EmptyVector if this Vector is empty.
         **/
        template <class ScalerType>
        Vector<T> &operator+=(const ScalerType &scaler);

        /**
         * Perform subtraction with two Vectors.
         * @param other a Vector as the second operand.
         * @return a new Vector as the subtraction result.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty or the number of elements of
         * the second operand is not a factor of that of this Vector.
         **/
        template <class OtherType>
        auto Minus(const Vector<OtherType> &other) const;

        /**
         * Perform element-wise subtraction with a scaler.
         * @param scaler a scaler to be subtracted from each element of the Vector.
         * @return a Vector as the subtraction result.
         * @throw EmptyVector if the Vector is empty.
         **/
        template <class ScalerType>
        auto Minus(const ScalerType &scaler) const;

        /**
         * Perform subtraction with two Vectors. (See Vector::Minus)
         * @param other a Vector as the second operand.
         * @return a new Vector as the subtraction result.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty or the number of elements of
         * the second operand is not a factor of that of this Vector.
         **/
        template <class OtherType>
        auto operator-(const Vector<OtherType> &other) const;

        /**
         * Perform subtraction with a Vector and a scaler. (See Vector.Minus)
         * @param scaler a scaler to be subtracted from each element of the Vector.
         * @return a new Vector that is the subtraction result.
         * @throw EmptyVector if this Vector is empty.
         **/
        template <class ScalerType>
        auto operator-(const ScalerType &scaler) const;

        /**
         * Perform inplace subtraction with another Vector.
         * @param other a Vector to be subtracted.
         * @return the reference to this Vector.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty or the number of elements of
         * the second operand is not a factor of that of this Vector.
         **/
        template <class OtherType>
        Vector<T> &operator-=(const Vector<OtherType> &other);

        /**
         * Perform element-wise multiplication with a Vector and a scaler.
         * @param scaler a scaler to be multiplied with each element of the Vector.
         * @return a new Vector as the multiplication result.
         * @throw EmptyVector if this Vector is empty.
         **/
        template <class OtherType>
        auto Scale(const OtherType &scaler) const;

        /**
         * Perform element-wise multiplication with a Vector and a scaler. (See Vector::Scale)
         * @param scaler a scaler used to scale this Vector.
         * @return a new Vector as the multiplication result.
         * @throw EmptyVector if the Vector is empty.
         **/
        template <class OtherType>
        auto operator*(const OtherType &scaler) const;

        /**
         * Perform element-wise multiplication with two Vectors.
         * @param other a Vector.
         * @return a Vector that is the result of the multiplication.
         * @throw EmptyVector when this vector is empty.
         * @throw InvalidArgument if the second operand is empty or the dimension of the second operand
         * is not a factor of that of the first operand.
         **/
        template <class OtherType>
        auto operator*(const Vector<OtherType> &other) const;

        /**
         * Scale a Vector inplace.
         * @param scaler a scaler to be multiplied by each element of the Vector.
         * @return the reference to the Vector.
         * @throw EmptyVector when this vector is empty.
         **/
        Vector<T> &operator*=(const T &scaler);

        /**
         * Perform inplace element-wise Vector multiplication.
         * @param other a Vector as the second operand.
         * @return the reference to this Vector.
         * @throw EmptyVector if this vector is empty.
         * @throw InvalidArgument if the dimensions of the Vectors are different.
         **/
        Vector<T> &operator*=(const Vector<T> &other);

        /**
         * Divide each element of the Vector by a scaler.
         * @param scaler a scaler to divide each element of the Vector.
         * @return a new Vector as the division result.
         * @throw EmptyVector if the Vector is empty.
         * @throw DividedByZero if the scaler is 0.
         **/
        template <class OtherType>
        auto Divide(const OtherType &scaler) const;

        /**
         * Perform element-wise division with two Vectors.
         * @param vector a Vector as the second operand.
         * @return a new Vector as the division result.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty of the dimension of the first
         * operand is not a factor of that of the second operand.
         * @throw DividedByZero if one of the elements of the second operand is 0.
         **/
        template <class OtherType>
        auto Divide(const Vector<OtherType> &vector) const;

        /**
         * Divide each element of a Vector by a scaler. (See Vector::Divide)
         * @param scaler a scaler used to divide each element of the Vector.
         * @return a new Vector as the division result.
         * @throw EmptyVector if the Vector is empty.
         * @throw DividedByZero if the scaler is 0.
         **/
        template <class OtherType>
        auto operator/(const OtherType &scaler) const;

        /**
         * Perform element-wise division with two Vectors. (See Vector::Divide)
         * @param vector a Vector as the second operand.
         * @return a new Vector as the division result.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty of the dimension of the first
         * operand is not a factor of that of the second operand.
         * @throw DividedByZero if one of the elements of the second operand is 0.
         **/
        template <class OtherType>
        auto operator/(const Vector<OtherType> &vector) const;

        /**
         * Perform inplace element-wise division with a Vector and a scaler.
         * @param scaler a scaler to divide each element of the Vector.
         * @return a reference to this Vector.
         * @throw EmptyVector if this Vector is empty.
         * @throw DividedByZero if the scaler is 0.
         **/
        template <class OtherType>
        Vector<T> &operator/=(const OtherType &scaler);

        /**
         * Perform inplace element-wise division.
         * @param vector a Vector as the second operand.
         * @return a reference to this Vector.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty of the dimension of the first
         * operand is not a factor of that of the second operand.
         * @throw DividedByZero if one of the elements of the second operand is 0.
         **/
        template <class OtherType>
        Vector<T> &operator/=(const Vector<OtherType> &vector);

        /**
         * Perform dot product with two Vectors.
         * @param other a Vector as the second operand.
         * @return a scaler as the dot product result.
         * @throw EmptyVector if this Vector is empty.
         * @throw InvalidArgument if the second operand is empty or the Vectors have different
         * dimensions.
         **/
        template <class OtherType>
        auto Dot(const Vector<OtherType> &other) const;

        /**
         * Return the normalized Vector.
         * @return a new Vector with the normalized values.
         * @throw EmptyVector if the Vector is empty.
         * @throw DividedByZero if this Vector is a zero vector.
         **/
        Vector<T> Normalized() const;

        /**
         * Normalize the Vector inplace.
         * @throw EmptyVector if the Vector is empty.
         * @throw DividedByZero if this Vector is a zero Vector.
         **/
        void Normalize();

        /**
         * Calculate the sum of all the elements of the Vector.
         * @return the sum of the elements.
         **/
        T Sum() const;

        /**
         * Map each element of the Vector to a new value.
         * @param f a function that maps the value of an element to a new value.
         * @return a new Vector with the new values produced by f.
         **/
        template <class MapFunction>
        auto Map(MapFunction &&f) const;

        /**
         * Return a pointer that points to the first element of the Vector.
         * @return a pointer pointing to the first element of the Vector.
         **/
        const T *AsRawPointer() const;

        /**
         * Generate a Vector filled with zeros.
         * @param n the number of zeros.
         * @return a Vector that is filled with n zeros.
         **/
        static Vector<T> ZeroVector(const std::size_t &n);

        /**
         * Stack all the elements of multiple Vectors in a new Vector.
         * @param vectors a std::initializer_list of Vectors to be stacked.
         * @return a Vector with the stacked elements.
         **/
        static Vector<T> Combine(const std::initializer_list<Vector<T>> &vectors);

        template <class ScalerType>
        friend auto operator+(const ScalerType &scaler, const Vector<T> &v);

        template <class ScalerType>
        friend auto operator-(const ScalerType &scaler, const Vector<T> &v);

        template <class ScalerType>
        friend auto operator*(const ScalerType &scaler, const Vector<T> &v);

        template <class ScalerType>
        friend auto operator/(const ScalerType &scaler, const Vector<T> &v);

        template <class OtherType>
        friend class Vector;
    };
}

#include "Vector.tpp"

#endif