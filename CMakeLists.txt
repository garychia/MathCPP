cmake_minimum_required(VERSION 3.14)

project(MathCPP VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

include(FetchContent)
FetchContent_Declare(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(gtest)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/MachineLearning)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/Misc)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)

add_executable(MathCPP application.cpp)
target_include_directories(
    MathCPP PUBLIC
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