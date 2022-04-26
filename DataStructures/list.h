#ifndef LIST_H
#define LIST_H

#define GENERATE_STRUCT_NAME(class_name, template_type) \
    class_name##_##template_type

#define DEFINE_STRUCT(class_name, template_type) \
    typedef struct GENERATE_STRUCT_NAME(class_name, template_type) GENERATE_STRUCT_NAME(class_name, template_type)

#define GENERATE_FUNCTION_SIGNATURE(template_type, return_type, func_name, class_name, ...) \
    return_type func_name##_##class_name##_##template_type(__VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_STRUCT(List, int);

/* Create a new empty List. */
GENERATE_FUNCTION_SIGNATURE(int, GENERATE_STRUCT_NAME(List, int)*, create, List);
/* Create a new List filled with the same value. */
GENERATE_FUNCTION_SIGNATURE(int, GENERATE_STRUCT_NAME(List, int)*, create, List, int n, int value);
/* Clone another List. */
GENERATE_FUNCTION_SIGNATURE(int, GENERATE_STRUCT_NAME(List, int)*, copy, List, const GENERATE_STRUCT_NAME(List, int)*);

#ifdef __cplusplus
}
#endif
#endif