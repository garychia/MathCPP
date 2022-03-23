#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>
#include <sstream>

namespace Exceptions
{
    class IndexOutOfBound : public std::exception
    {
    public:
        IndexOutOfBound(const std::size_t &index, const std::string& additionalMessage = "") : invalidIndex(index)
        {
            std::stringstream ss;
            ss << "Index Out of Bound: " << invalidIndex;
            if (additionalMessage.size() > 0)
                ss << std::endl << additionalMessage;
            errorMessage = ss.str();
        }

        const char *what() const throw()
        {
            return errorMessage.c_str();
        }

    private:
        std::size_t invalidIndex;
        std::string errorMessage;
    };
}

#endif
