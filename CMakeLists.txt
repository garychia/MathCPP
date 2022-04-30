cmake_minimum_required(VERSION 3.10)

project(MathCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

include_directories(
    ${PROJECT_SOURCE_DIR}/Algorithms
    ${PROJECT_SOURCE_DIR}/DataStructures
    ${PROJECT_SOURCE_DIR}/Exceptions)

file(GLOB SOURCES CONFIGURE_DEPENDS "DataStructures/*.hpp")

add_library(mathcpp_data_structures SHARED ${SOURCES})
set_target_properties(mathcpp_data_structures PROPERTIES LINKER_LANGUAGE CXX)

add_executable(MathCPP application.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MathCPP PUBLIC OpenMP::OpenMP_CXX)
endif()