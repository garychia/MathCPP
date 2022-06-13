#include <gtest/gtest.h>
#include "tuple_test.hpp"
#include "vector_test.hpp"
#include "vector3d_test.hpp"

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}