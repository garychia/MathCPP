#ifndef MATRIX_H
#define MATRIX_H

#include <sstream>

#include "container.hpp"
#include "vector.hpp"
#include "exceptions.hpp"

namespace Math
{
    template <class T>
    class Matrix : public Container<Vector<T>>
    {
    private:
        Vector<Vector<T>> rows;
        std::size_t nRows;
        std::size_t nColumns;

    public:
        /*
        Constructor that creates an empty matrix.
        */
        Matrix() : rows(), nRows(0), nColumns(0) {}

        /*
        Constructor with an initializer_list of row Vectors.
        @param l an initializer_list that contains row Vectors.
        @throw DimensionMismatch when there is any of the row Vectors has a
        different dimension.
        */
        Matrix(std::initializer_list<Vector<T>> l) : rows(l)
        {
            if (l.size() > 0)
            {
                nRows = l.size();
                nColumns = l.begin()->Size();
                for (auto itr = l.begin() + 1; itr != l.end(); itr++)
                {
                    if (itr->Dimension() != nColumns)
                        throw Exceptions::DimensionMismatch(
                            nColumns,
                            itr->Dimension(),
                            "Matrix: All row vectors must be of the same dimension.");
                }
            }
            else
            {
                nRows = 0;
                nColumns = 0;
            }
        }

        /*
        Accesses the vector at a given index.
        @return the vector at the given index.
        @throw IndexOutOfBound when the index exceeds the greatest possible index.
        */
        virtual Vector<T> &operator[](const std::size_t &index)
        {
            try
            {
                return rows[index];
            }
            catch (const Exceptions::IndexOutOfBound &)
            {
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Matrix: Index must be less than the number of rows.");
            }
        }

        /*
        Accesses the vector at a given index.
        @return the vector at the given index.
        @throw IndexOutOfBound when the index exceeds the greatest possible index.
        */
        virtual const Vector<T> &operator[](const std::size_t &index) const override
        {
            try
            {
                return rows[index];
            }
            catch (const Exceptions::IndexOutOfBound &)
            {
                throw Exceptions::IndexOutOfBound(
                    index,
                    "Matrix: Index must be less than the number of rows.");
            }
        }

        /*
        Returns the number of row Vectors this Matrix stores.
        @return the number of row Vectors this Matrix stores.
        */
        virtual std::size_t Size() const override
        {
            return rows.Size();
        };

        /*
        Converts this Matrix to a string that shows the elements of this Matrix.
        @return a string that represents this Matrix.
        */
        virtual std::string ToString() const override
        {
            if (nRows == 0)
                return "";
            std::stringstream ss;
            for (std::size_t i = 0; i < nRows; i++)
            {
                ss << rows[i];
                if (i < nRows - 1)
                    ss << std::endl;
            }
            return ss.str();
        }

        /*
        Converts this Matrix to a string and pass it to a output stream.
        @param os an output stream.
        @param m a Matrix
        @return the output stream.
        */
        friend std::ostream &operator<<(std::ostream& os, const Matrix<T>& m)
        {
            os << m.ToString();
            return os;
        }
    };
} // namespace Math

#endif