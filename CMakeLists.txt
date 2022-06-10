cmake_minimum_required(VERSION 3.10)

project(MathCPP VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

add_subdirectory(MachineLearning)
add_subdirectory(Misc)

add_executable(MathCPP application.cpp)
target_include_directories(MathCPP PUBLIC
                          "${PROJECT_SOURCE_DIR}/Algorithms"
                          "${PROJECT_SOURCE_DIR}/DataStructures"
                          "${PROJECT_SOURCE_DIR}/Exceptions"
                          "${PROJECT_SOURCE_DIR}/MachineLearning"
                          "${PROJECT_SOURCE_DIR}/Misc"
                          )

target_link_libraries(MathCPP PUBLIC MathCPPML)
target_link_libraries(MathCPP PUBLIC Misc)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MathCPP PUBLIC OpenMP::OpenMP_CXX)
endif()