cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME MathCPP)

project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    add_compile_definitions(__CUDA_ENABLED__=1)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

include(FetchContent)
FetchContent_Declare(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(gtest)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/Algorithms)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/MachineLearning)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/Misc)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/MathCUDA)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)

set(DATASTRUCTURES_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/DataStructures/ComputationGraph.hpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/ComputationGraph.tpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/Container.hpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/List.hpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/List.tpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/Matrix.hpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/ScalerComputationGraph.hpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/ScalerComputationGraph.tpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/Tuple.hpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/Vector.hpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/Vector3D.hpp
    ${PROJECT_SOURCE_DIR}/src/DataStructures/Vector3D.tpp)

add_executable(MathCPP application.cpp ${DATASTRUCTURES_SOURCE_FILES})
target_include_directories(
    MathCPP PUBLIC
    ${PROJECT_SOURCE_DIR}/src/Algorithms
    ${PROJECT_SOURCE_DIR}/src/DataStructures
    ${PROJECT_SOURCE_DIR}/src/Exceptions
    ${PROJECT_SOURCE_DIR}/src/MachineLearning
    ${PROJECT_SOURCE_DIR}/src/Misc
    ${PROJECT_SOURCE_DIR}/src/MathCUDA
)

if (MSVC)
    target_compile_options(Misc PUBLIC "/openmp:llvm")
endif (MSVC)

target_link_libraries(MathCPP PUBLIC MathCPPMath)
target_link_libraries(MathCPP PUBLIC MathCPPNN)
target_link_libraries(MathCPP PUBLIC Misc)
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(MathCPP PUBLIC MathCPPCUDA)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MathCPP PUBLIC OpenMP::OpenMP_CXX)
endif()