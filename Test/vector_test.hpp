#include <gtest/gtest.h>

#include "../vector.hpp"

using namespace DataStructure;

TEST(Vector, VectorConstructor)
{
    Vector<int> emptyVector;
    EXPECT_EQ(emptyVector.Size(), 0);
    EXPECT_EQ(emptyVector.Dimension(), 0);
}

TEST(Vector, ZeroVector)
{
    const int length = 50000000;
    auto zeroVector = Vector<int>::ZeroVector(length);
    for (int i = 0; i < length; i++)
        EXPECT_EQ(zeroVector[i], 0);
}