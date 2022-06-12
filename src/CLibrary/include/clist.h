#ifndef LIST_H
#define LIST_H

#include "clib_macros.h"

#ifdef __cplusplus
extern "C"
{
#endif

    DEFINE_STRUCT(List, int);

    /* Create a new empty List. */
    GENERATE_FUNCTION_SIGNATURE(int, GENERATE_STRUCT_NAME(List, int) *, create, List);
    /* Create a new List filled with the same value. */
    GENERATE_FUNCTION_SIGNATURE(int, GENERATE_STRUCT_NAME(List, int) *, create_with_same_values, List, int n, int value);
    /* Clone another List. */
    GENERATE_FUNCTION_SIGNATURE(int, GENERATE_STRUCT_NAME(List, int) *, copy, List, const GENERATE_STRUCT_NAME(List, int) * list);
    /* Destroy a list. */
    GENERATE_FUNCTION_SIGNATURE(int, void, destroy, List, GENERATE_STRUCT_NAME(List, int) * list);
    /* Retrieve an element from a list. */
    GENERATE_FUNCTION_SIGNATURE(int, int, get, List, const GENERATE_STRUCT_NAME(List, int) * list, unsigned long index);

#ifdef __cplusplus
}
#endif

#endif // LIST_H