set(MISC_SOURCES ${PROJECT_SOURCE_DIR}/Misc/Random.cpp)
set(MISC_HEADERS ${PROJECT_SOURCE_DIR}/Misc/Random.hpp)

add_library(Misc ${MISC_SOURCES} ${MISC_HEADERS})
target_include_directories(Misc PUBLIC
    ${PROJECT_SOURCE_DIR}/Algorithms
    ${PROJECT_SOURCE_DIR}/DataStructures
    ${PROJECT_SOURCE_DIR}/Exceptions
    ${PROJECT_SOURCE_DIR}/MachineLearning
    ${PROJECT_SOURCE_DIR}/Misc)