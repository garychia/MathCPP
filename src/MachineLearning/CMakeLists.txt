set(ML_SOURCES
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/Algorithms.tpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/LinearLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/ActivationLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/TanhLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/ReLULayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/SoftMaxLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/LossLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NLLLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork.cpp)

set(ML_HEADERS
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/Algorithms.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/LinearLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/ActivationLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/TanhLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/ReLULayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/SoftMaxLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/LossLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NLLLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork.hpp)

add_library(MathCPPML ${ML_SOURCES} ${ML_HEADERS})
target_include_directories(MathCPPML PUBLIC
        ${CMAKE_SOURCE_DIR}/src/Algorithms
        ${CMAKE_SOURCE_DIR}/src/DataStructures
        ${CMAKE_SOURCE_DIR}/src/Exceptions
        ${CMAKE_SOURCE_DIR}/src/MachineLearning
        ${CMAKE_SOURCE_DIR}/src/Misc
        ${CMAKE_SOURCE_DIR}/src/MathCUDA)

if (MSVC)
    target_compile_options(MathCPPML PUBLIC "/openmp:llvm")
endif (MSVC)
