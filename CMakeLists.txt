cmake_minimum_required(VERSION 3.10)

project(MathCPP VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

add_executable(MathCPP application.cpp)

target_include_directories(MathCPP PUBLIC
                          "${PROJECT_SOURCE_DIR}/Algorithms"
                          "${PROJECT_SOURCE_DIR}/DataStructures"
                          "${PROJECT_SOURCE_DIR}/Exceptions"
                          )

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MathCPP PUBLIC OpenMP::OpenMP_CXX)
endif()