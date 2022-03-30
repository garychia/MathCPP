cmake_minimum_required(VERSION 3.10)

project(MathCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

include_directories(
    ${PROJECT_SOURCE_DIR}/Algorithms
    ${PROJECT_SOURCE_DIR}/DataStructures
    ${PROJECT_SOURCE_DIR}/Exceptions)
add_executable(MathCPP application.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MathCPP PUBLIC OpenMP::OpenMP_CXX)
endif()
