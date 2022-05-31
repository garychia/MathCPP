#include "clib_macros.h"
#include "list.hpp"

using namespace DataStructure;

extern "C"
{

    typedef List<int> GENERATE_STRUCT_NAME(List, int);

    /* Create a new empty List. */
    GENERATE_FUNCTION_SIGNATURE(int, GENERATE_STRUCT_NAME(List, int) *, create, List)
    {
        return new List<int>();
    }
    /* Create a new List filled with the same value. */
    GENERATE_FUNCTION_SIGNATURE(int, GENERATE_STRUCT_NAME(List, int) *, create_with_same_values, List, int n, int value)
    {
        return new List<int>(n, value);
    }
    /* Clone another List. */
    GENERATE_FUNCTION_SIGNATURE(int, GENERATE_STRUCT_NAME(List, int) *, copy, List, const GENERATE_STRUCT_NAME(List, int) * list)
    {
        return new List<int>(*list);
    }

    GENERATE_FUNCTION_SIGNATURE(int, void, destroy, List, GENERATE_STRUCT_NAME(List, int) * list)
    {
        if (list)
        {
            delete list;
            list = nullptr;
        }
    }

    GENERATE_FUNCTION_SIGNATURE(int, int, get, List, const GENERATE_STRUCT_NAME(List, int)* list, unsigned long index)
    {
        return (*list)[index];
    }
}