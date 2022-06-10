set(ML_SOURCES
    ${CMAKE_SOURCE_DIR}/MachineLearning/Algorithms.tpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/NeuralLayer.cpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/LinearLayer.cpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/ActivationLayer.cpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/TanhLayer.cpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/ReLULayer.cpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/SoftMaxLayer.cpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/NeuralNetwork.cpp)

set(ML_HEADERS
    ${CMAKE_SOURCE_DIR}/MachineLearning/Algorithms.hpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/NeuralLayer.hpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/LinearLayer.hpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/ActivationLayer.hpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/TanhLayer.hpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/ReLULayer.hpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/SoftMaxLayer.hpp
    ${CMAKE_SOURCE_DIR}/MachineLearning/NeuralNetwork.hpp)

add_library(MathCPPML ${ML_SOURCES} ${ML_HEADERS})
target_include_directories(MathCPPML PUBLIC
    "${PROJECT_SOURCE_DIR}/Algorithms"
    "${PROJECT_SOURCE_DIR}/DataStructures"
    "${PROJECT_SOURCE_DIR}/Exceptions"
    "${PROJECT_SOURCE_DIR}/MachineLearning"
    "${PROJECT_SOURCE_DIR}/Misc")