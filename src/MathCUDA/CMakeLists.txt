if(CMAKE_CUDA_COMPILER)
    set(CUDA_SOURCES
        ${PROJECT_SOURCE_DIR}/src/MathCUDA/ArithmeticOperations.cu
        ${PROJECT_SOURCE_DIR}/src/MathCUDA/CUDAHelpers.cu
        ${PROJECT_SOURCE_DIR}/src/MathCUDA/CUDAUtilities.cu)
    set(CUDA_HEADERS
        ${PROJECT_SOURCE_DIR}/src/MathCUDA/ArithmeticOperations.cuh
        ${PROJECT_SOURCE_DIR}/src/MathCUDA/CUDAHelpers.hpp
        ${PROJECT_SOURCE_DIR}/src/MathCUDA/CUDAUtilities.cuh)
    add_library(MathCPPCUDA STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})
    include_directories(
        ${PROJECT_SOURCE_DIR}/src/Algorithms
        ${PROJECT_SOURCE_DIR}/src/DataStructures
        ${PROJECT_SOURCE_DIR}/src/Exceptions
        ${PROJECT_SOURCE_DIR}/src/Misc)
    find_package(CUDAToolkit)
    target_link_libraries(MathCPPCUDA PRIVATE CUDA::cudart)
    set_target_properties(MathCPPCUDA PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_options(MathCPPCUDA PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)
endif()