cmake_minimum_required(VERSION 3.10)

project(MathCPP LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

include_directories(
    ${PROJECT_SOURCE_DIR}/Algorithms
    ${PROJECT_SOURCE_DIR}/DataStructures
    ${PROJECT_SOURCE_DIR}/Exceptions)

file(GLOB CXX_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/DataStructures/*.hpp
    ${PROJECT_SOURCE_DIR}/DataStructures/*.cpp)

file(GLOB C_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/DataStructures/c_lib/*.h
    ${PROJECT_SOURCE_DIR}/DataStructures/c_lib/*.cpp)

add_library(MathCPP SHARED ${CXX_SOURCES})
set_target_properties(MathCPP PROPERTIES LINKER_LANGUAGE CXX)
add_library(MathCPPC SHARED ${C_SOURCES})
set_target_properties(MathCPPC PROPERTIES LINKER_LANGUAGE CXX)

add_executable(MathCPPApplication application.cpp)
add_executable(MathCPPApplicationInC c_main.c)
target_link_libraries(MathCPPApplicationInC MathCPPC MathCPP)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MathCPP PUBLIC OpenMP::OpenMP_CXX)
endif()