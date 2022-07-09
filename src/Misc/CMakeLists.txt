set(MISC_SOURCES ${PROJECT_SOURCE_DIR}/src/Misc/Random.cpp)
set(MISC_HEADERS ${PROJECT_SOURCE_DIR}/src/Misc/Random.hpp)

add_library(Misc ${MISC_SOURCES} ${MISC_HEADERS})
target_include_directories(Misc PUBLIC
    ${PROJECT_SOURCE_DIR}/src/Algorithms
    ${PROJECT_SOURCE_DIR}/src/DataStructures
    ${PROJECT_SOURCE_DIR}/src/Exceptions
    ${PROJECT_SOURCE_DIR}/src/MachineLearning
    ${PROJECT_SOURCE_DIR}/src/Misc
    ${PROJECT_SOURCE_DIR}/src/MathCUDA)

if (MSVC)
    target_compile_options(Misc PUBLIC "/openmp:llvm")
endif (MSVC)