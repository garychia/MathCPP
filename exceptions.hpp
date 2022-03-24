#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>
#include <sstream>

namespace Exceptions
{

    class Exception : public std::exception
    {
    protected:
        // an error message to be displayed.
        std::string errorMessage;

    public:
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
        DividedByZero(std::string additionalMessage = "")
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
        /*
        Constructor with an Expected Dimension, a Mismatched Dimension, and Optional Message
        @param expectedDimension the expected dimension
        @param misMatchedDimension the mismatched dimension
        @param additionalMessage a string that represent an optional error message.
        */
        DimensionMismatch(
            std::size_t expectedDimension,
            std::size_t misMatchedDimension,
            std::string additionalMessage = "")
            : expectedDimension(expectedDimension),
              misMatchedDimension(misMatchedDimension)
        {
            std::stringstream ss;
            ss << "Mismatched dimension found: "
               << misMatchedDimension << std::endl;
            ss << "Expected dimension: "
               << expectedDimension << std::endl;
            if (additionalMessage.size() > 0)
                ss << std::endl
                   << additionalMessage;
            errorMessage = ss.str();
        }

    private:
        // the expected dimension
        std::size_t expectedDimension;
        // the found mismatched dimension
        std::size_t misMatchedDimension;
    };
}

#endif
