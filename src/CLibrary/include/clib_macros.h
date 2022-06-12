#define GENERATE_STRUCT_NAME(class_name, template_type) \
    class_name##_##template_type

#define DEFINE_STRUCT(class_name, template_type) \
    typedef struct GENERATE_STRUCT_NAME(class_name, template_type) GENERATE_STRUCT_NAME(class_name, template_type)

#define GENERATE_FUNCTION_SIGNATURE(template_type, return_type, func_name, class_name, ...) \
    return_type func_name##_##class_name##_##template_type(__VA_ARGS__)