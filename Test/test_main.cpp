#include <gtest/gtest.h>
#include "math_test.hpp"
#include "tuple_test.hpp"
#include "vector_test.hpp"
#include "matrix_test.hpp"

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}