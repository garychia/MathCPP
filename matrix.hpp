#ifndef MATRIX_H
#define MATRIX_H

#include <sstream>

#include "container.hpp"
#include "vector.hpp"
#include "list.hpp"
#include "exceptions.hpp"

namespace DataStructure
{
    template <class T>
    class Matrix : public Container<Vector<T>>
    {
    private:
        // the number of row vectors
        std::size_t nRows;
        // the number of columns of each row vector.
        std::size_t nColumns;

    public:
        /*
        Constructor that creates an empty matrix.
        */
        Matrix() : Container<Vector<T>>(), nRows(0), nColumns(0) {}

        /*
        Constructor with the Number of Rows and Columns Specified.
        @param numRows the number of rows.
        @param numCoumns the number of columns.
        @param value the value, 0 by default, the Matrix will be filled with.
        */
        Matrix(std::size_t numRows, std::size_t numColumns, T initialValue = 0)
        {
            nRows = numRows;
            nColumns = numColumns;
            this->size = numRows * numColumns;
            this->data = nullptr;
            if (numRows > 0)
            {
                if (numColumns == 0)
                    throw Exceptions::InvalidArgument(
                        "Matrix: the number of columns must be a positive number for a non-empty matrix.");
                
                this->data = new Vector<T>[numRows];
            
                std::size_t i;
                #pragma omp parallel for schedule(dynamic)
                for (i = 0; i < numRows; i++)
                    this->data[i] = Vector<T>::ZeroVector(numColumns);
                
                if (initialValue != 0)
                {
                    std::size_t j;
                    #pragma omp parallel for private(j) schedule(dynamic) collapse(2)
                    for (i = 0; i < numRows; i++)
                        for (j = 0; j < numColumns; j++)
                            this->data[i][j] = initialValue;
                }
            }
            else if (numColumns > 0)
            {
                throw Exceptions::InvalidArgument(
                    "Matrix: Cannot initialize a matrix with no row vector while the number of columns is greater than 0.");
            }
        }

        /*
        Constructor with an initializer_list of row Vectors.
        @param l an initializer_list that contains row Vectors.
        @throw DimensionMismatch when there is any of the row Vectors has a
        different dimension.
        */
        Matrix(std::initializer_list<Vector<T>> l)
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
                Container<Vector<T>>::Container(l);
                this->size = nRows * nColumns;
            }
            else
            {
                nRows = 0;
                nColumns = 0;
                this->size = 0;
                this->data = nullptr;
            }
        }

        /*
        Constructor with arrary as Input.
        @param arr an array that contains the row vectors this Matrix will store.
        */
        template <std::size_t N>
        Matrix(const std::array<Vector<T>, N>& arr)
        {
            if (arr.size() > 0)
            {
                nRows = arr.size();
                nColumns = arr.begin()->Size();
                this->size = nRows * nColumns;
                for (std::size_t i = 0; i < nRows; i++)
                {
                    if (arr[i].Dimension() != nColumns)
                        throw Exceptions::DimensionMismatch(
                            nColumns,
                            arr[i].Dimension(),
                            "Matrix: All row vectors must be of the same dimension.");
                }
                Container<Vector<T>>::Container(arr);
            }
            else
            {
                nRows = 0;
                nColumns = 0;
                this->size = 0;
                this->data = nullptr;
            }
        }

        /*
        Constructor with List as Input.
        @param l a List that contains the row vectors this Matrix will store.
        */
        Matrix(const List<Vector<T>>& l)
        {
            const std::size_t listSize = l.Size();
            if (listSize > 0)
            {
                nRows = listSize;
                nColumns = l[0].Size();
                this->size = nRows * nColumns;
                #pragma omp parallel for schedule(dynamic)
                for (std::size_t i = 0; i < nRows; i++)
                {
                    if (l[i].Dimension() != nColumns)
                        throw Exceptions::DimensionMismatch(
                            nColumns,
                            l[i].Dimension(),
                            "Matrix: All row vectors must be of the same dimension.");
                }
                this->data = new Vector<T>[nRows];
                #pragma omp parallel for schedule(dynamic)
                for (std::size_t i = 0; i < nRows; i++)
                    this->data[i] = l[i];
            }
            else
            {
                nRows = 0;
                nColumns = 0;
                this->size = 0;
                this->data = nullptr;
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
                return this->data[index];
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
                return this->data[index];
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
        virtual std::size_t Size() const override { return this->size; }

        /*
        Returns the shape of this Matrix.
        @return a Tuple that contains the numbers of rows and columns.
        */
        Tuple<std::size_t> Shape() const { return Tuple<std::size_t>({nRows, nColumns}); }

        /*
        Converts this Matrix to a string that shows the elements of this Matrix.
        @return a string that represents this Matrix.
        */
        virtual std::string ToString() const override
        {
            if (nRows == 0)
                return "(EMPTY MATRIX)";
            std::stringstream ss;
            for (std::size_t i = 0; i < nRows; i++)
            {
                ss << this->data[i];
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
        friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &m)
        {
            os << m.ToString();
            return os;
        }

        /*
        Transposes this Matrix inplace.
        */
        void Transpose()
        {
            if (nRows > 0)
            {
                std::size_t i, j;
                Vector<T>* newRows = new Vector<T>[nColumns];
                #pragma omp parallel for schedule(dynamic)
                for (i = 0; i < nColumns; i++)
                    newRows[i] = Vector<T>::ZeroVector(nRows);
                #pragma omp parallel for private(j) schedule(dynamic) collapse(2)
                for (i = 0; i < nColumns; i++)
                    for (j = 0; j < nRows; j++)
                        newRows[i][j] = this->data[j][i];
                delete[] this->data;
                this->data = newRows;
                std::swap(nRows, nColumns);
            }
        }
    };
} // namespace Math

#endif