#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>
#include <sstream>

namespace DataStructure
{
    template <class T>
    class Tuple;
}

namespace Exceptions
{
    class Exception : public std::exception
    {
    protected:
        // an error message to be displayed.
        std::string errorMessage;

    public:
        /* Constructor */
        Exception(const std::string &message = "") : errorMessage(message) {}

        /*
        Returns the error message.
        @return an error message.
        */
        virtual const char *what() const throw()
        {
            return errorMessage.c_str();
        }
    };

    /*
    IndexOutOfBound is an exception that is thrown when an index is out of bound.
    */
    class IndexOutOfBound : public Exception
    {
    public:
        /*
        Constructor with a detected invalid index and an option message to be shown.
        @param index a detected invalid index.
        @param addtionalMessage an optional string that represents an error message.
        */
        IndexOutOfBound(const std::size_t &index, const std::string &additionalMessage = "") : invalidIndex(index)
        {
            std::stringstream ss;
            ss << "Index Out of Bound: " << invalidIndex;
            if (additionalMessage.size() > 0)
                ss << std::endl
                   << additionalMessage;
            errorMessage = ss.str();
        }

    private:
        // a detected invalid index.
        std::size_t invalidIndex;
    };

    /*
    DividedByZero is an exception thown when a value is being divided by zero.
    */
    class DividedByZero : public Exception
    {
    public:
        /*
        Constructor with an optional message.
        @param additionalMessage a string that represent an optional error message.
        */
        DividedByZero(const std::string &additionalMessage = "")
        {
            std::stringstream ss;
            ss << "Division by zero occurred.";
            if (additionalMessage.size() > 0)
                ss << std::endl
                   << additionalMessage;
            errorMessage = ss.str();
        }
    };

    /*
    DimensionMismatch is an exception thown when a mismatched dimension of
    a vector is found.
    */
    class DimensionMismatch : public Exception
    {
    public:
        /*
        Constructor with an Expected Dimension, a Mismatched Dimension, and Optional Message
        @param expectedDimension the expected dimension
        @param misMatchedDimension the mismatched dimension
        @param additionalMessage a string that represent an optional error message.
        */
        DimensionMismatch(
            std::size_t expectedDimension,
            std::size_t misMatchedDimension,
            const std::string &additionalMessage = "")
        {
            std::stringstream ss;
            ss << "Mismatched dimension found: "
               << misMatchedDimension << std::endl;
            ss << "Expected dimension: "
               << expectedDimension;
            if (additionalMessage.size() > 0)
                ss << std::endl
                   << additionalMessage;
            errorMessage = ss.str();
        }
    };

    class EmptyVector : public Exception
    {
    public:
        EmptyVector(const std::string &message = "") : Exception(message) {}
    };

    class EmptyMatrix : public Exception
    {
    public:
        EmptyMatrix(const std::string &message = "") : Exception(message) {}
    };

    class MatrixShapeMismatch : Exception
    {
    public:
        template <class T>
        MatrixShapeMismatch(
            const DataStructure::Tuple<T> &matrixShape,
            const DataStructure::Tuple<T> &targetShape,
            const std::string &message = "")
        {
            std::stringstream errorMessageStream;
            errorMessageStream
                << "Matrix Shape: " << matrixShape << std::endl
                << "Target Shape: " << targetShape << std::endl;
            if (message.length() > 0)
                errorMessageStream << message << std::endl;
            this->errorMessage = errorMessageStream.str();
        }

        template <class T>
        MatrixShapeMismatch(
            const DataStructure::Tuple<T> &expectedShape = DataStructure::Tuple<T>(),
            const std::string &message = "")
        {
            std::stringstream ss;
            if (expectedShape.Size() > 0)
            {
                ss << "Expected Shape: "
                   << expectedShape << std::endl;
            }
            ss << message;
            errorMessage = ss.str();
        }
    };

    class EmptyList : public Exception
    {
    public:
        EmptyList(const std::string &message = "") : Exception(message) {}
    };

    class InvalidArgument : public Exception
    {
    public:
        InvalidArgument(const std::string &message = "") : Exception(message) {}
    };

    class NoImplementation : public Exception
    {
    public:
        NoImplementation(const std::string &message = "") : Exception(message) {}
    };

    class GradientNotEvaluated : public Exception
    {
    public:
        GradientNotEvaluated(const std::string &message = "") : Exception(message) {}
    };
}

#endif
