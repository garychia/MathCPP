enable_testing()
add_executable(MathCPPTest test_main.cpp)
target_include_directories(
    MathCPPTest PUBLIC
    ${PROJECT_SOURCE_DIR}/src/Algorithms
    ${PROJECT_SOURCE_DIR}/src/DataStructures
    ${PROJECT_SOURCE_DIR}/src/Exceptions
    ${PROJECT_SOURCE_DIR}/src/MachineLearning
    ${PROJECT_SOURCE_DIR}/src/Misc
    ${PROJECT_SOURCE_DIR}/test
)

target_link_libraries(
    MathCPPTest
    gtest_main
)
target_include_directories(
    MathCPPTest PUBLIC
    ${PROJECT_SOURCE_DIR}/test/DataStructures
)

include(GoogleTest)
gtest_discover_tests(MathCPPTest)