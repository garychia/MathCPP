#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>
#include <sstream>

namespace Exceptions
{
    /*
    IndexOutOfBound is an exception that is thrown when an index is out of bound.
    */
    class IndexOutOfBound : public std::exception
    {
    public:
        /*
        Constructor with a detected invalid index and an option message to be shown.
        @param index a detected invalid index.
        @param addtionalMessage an optional string that represents an error message.
        */
        IndexOutOfBound(const std::size_t &index, const std::string& additionalMessage = "") : invalidIndex(index)
        {
            std::stringstream ss;
            ss << "Index Out of Bound: " << invalidIndex;
            if (additionalMessage.size() > 0)
                ss << std::endl << additionalMessage;
            errorMessage = ss.str();
        }

        /*
        Returns the error message.
        @return an error message.
        */
        const char *what() const throw()
        {
            return errorMessage.c_str();
        }

    private:
        // a detected invalid index.
        std::size_t invalidIndex;
        // an error message to be shown.
        std::string errorMessage;
    };
}

#endif
