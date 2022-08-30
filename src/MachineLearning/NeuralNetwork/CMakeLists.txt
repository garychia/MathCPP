set(NN_SOURCES
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/Algorithms.tpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/NeuralLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/LinearLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/ActivationLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/TanhLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/ReLULayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/SoftMaxLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/LossLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/NLLLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/BatchNormLayer.cpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/NeuralNetwork.cpp)

set(NN_HEADERS
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/Algorithms.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/NeuralLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/LinearLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/ActivationLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/TanhLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/ReLULayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/SoftMaxLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/LossLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/NLLLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/Layers/BatchNormLayer.hpp
    ${CMAKE_SOURCE_DIR}/src/MachineLearning/NeuralNetwork/NeuralNetwork.hpp)

add_library(MathCPPNN ${NN_SOURCES} ${NN_HEADERS})

if(CMAKE_CUDA_COMPILER)
    target_include_directories(
        MathCPPNN PUBLIC
        ${PROJECT_SOURCE_DIR}/src/Algorithms
        ${PROJECT_SOURCE_DIR}/src/DataStructures
        ${PROJECT_SOURCE_DIR}/src/Exceptions
        ${PROJECT_SOURCE_DIR}/src/MachineLearning
        ${PROJECT_SOURCE_DIR}/src/Misc
        ${PROJECT_SOURCE_DIR}/test/DataStructures
        ${PROJECT_SOURCE_DIR}/src/MathCUDA
    )
    target_link_libraries(MathCPPNN MathCPPCUDA)
else()
    target_include_directories(
        MathCPPNN PUBLIC
        ${PROJECT_SOURCE_DIR}/src/Algorithms
        ${PROJECT_SOURCE_DIR}/src/DataStructures
        ${PROJECT_SOURCE_DIR}/src/Exceptions
        ${PROJECT_SOURCE_DIR}/src/MachineLearning
        ${PROJECT_SOURCE_DIR}/src/Misc
        ${PROJECT_SOURCE_DIR}/test/DataStructures
    )
endif()

if (MSVC)
    target_compile_options(MathCPPNN PUBLIC "/openmp:llvm")
endif (MSVC)