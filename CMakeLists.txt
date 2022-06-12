cmake_minimum_required(VERSION 3.10)

project(MathCPP VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

add_subdirectory(${PROJECT_SOURCE_DIR}/src/MachineLearning)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/Misc)

add_executable(MathCPP application.cpp)
target_include_directories(MathCPP PUBLIC
                          ${PROJECT_SOURCE_DIR}/src/Algorithms
                          ${PROJECT_SOURCE_DIR}/src/DataStructures
                          ${PROJECT_SOURCE_DIR}/src/Exceptions
                          ${PROJECT_SOURCE_DIR}/src/MachineLearning
                          ${PROJECT_SOURCE_DIR}/src/Misc
                          )

target_link_libraries(MathCPP PUBLIC MathCPPML)
target_link_libraries(MathCPP PUBLIC Misc)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MathCPP PUBLIC OpenMP::OpenMP_CXX)
endif()