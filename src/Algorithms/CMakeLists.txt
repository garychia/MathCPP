set(MATH_SOURCE_FILES Math.hpp Math.cpp)

add_library(MathCPPMath STATIC ${MATH_SOURCE_FILES})

target_include_directories(
MathCPPMath PUBLIC
    ${PROJECT_SOURCE_DIR}/src/Algorithms
    ${PROJECT_SOURCE_DIR}/src/DataStructures
    ${PROJECT_SOURCE_DIR}/src/Exceptions
)
