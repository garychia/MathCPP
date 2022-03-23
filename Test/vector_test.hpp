#include <gtest/gtest.h>

#include "../vector.hpp"

using namespace Math;

TEST(Vector, VectorConstructor)
{
    Vector<int> emptyVector;
    EXPECT_EQ(emptyVector.Size(), 0);
    EXPECT_EQ(emptyVector.Dimension(), 0);
}