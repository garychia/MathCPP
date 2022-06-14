#include <gtest/gtest.h>

#include "Vector.hpp"

#define ZERO 0

using namespace DataStructures;

TEST(Vector, VectorEmptyConstructor)
{
    Vector<int> empty;
    EXPECT_EQ(empty.Size(), ZERO);
    EXPECT_EQ(empty.Dimension(), ZERO);
}

TEST(Vector, VectorFillConstructor)
{
    const int VECTOR_LENGHTS[] = {0, 2, 4, 8, 16, 32, 64, 128, 256};
    for (int i = 0; i < sizeof(VECTOR_LENGHTS) / sizeof(VECTOR_LENGHTS[ZERO]); i++)
    {
        // Fill the vector with its size (dimension).
        Vector<int> v(VECTOR_LENGHTS[i], VECTOR_LENGHTS[i]);
        EXPECT_EQ(v.Size(), VECTOR_LENGHTS[i]);
        EXPECT_EQ(v.Dimension(), VECTOR_LENGHTS[i]);
        for (int j = 0; j < VECTOR_LENGHTS[j]; j++)
            EXPECT_EQ(v[j], VECTOR_LENGHTS[i]);
    }
}

TEST(Vector, VectorInitializerListConstructor)
{
    const auto INT_VECTOR_CONTENT = {43, -13, 90, -39, 0, 23, -75};
    const auto FLOAT_VECTOR_CONTENT = {-2.124f, 23.2f - 82.32f, 2343.3f, 1.04f, 0.f, 321.3f, -9.f};
    const auto DOUBLE_VECTOR_CONTENT = {3.14, -1.234, -0.234, -94.3, 0.0, 23.0, -7.5, 0.85};
    Vector<int> initVectorInt(INT_VECTOR_CONTENT);
    Vector<float> initVectorFloat(FLOAT_VECTOR_CONTENT);
    Vector<double> initVectorDouble(DOUBLE_VECTOR_CONTENT);
    EXPECT_EQ(initVectorInt.Size(), INT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorFloat.Size(), FLOAT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorDouble.Size(), DOUBLE_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorInt.Dimension(), INT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorFloat.Dimension(), FLOAT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorDouble.Dimension(), DOUBLE_VECTOR_CONTENT.size());
    int i = 0;
    for (const auto &content : INT_VECTOR_CONTENT)
        EXPECT_EQ(initVectorInt[i++], content);
    i = 0;
    for (const auto &content : FLOAT_VECTOR_CONTENT)
        EXPECT_FLOAT_EQ(initVectorFloat[i++], content);
    i = 0;
    for (const auto &content : DOUBLE_VECTOR_CONTENT)
        EXPECT_DOUBLE_EQ(initVectorDouble[i++], content);
}

TEST(Vector, VectorArrayConstructor)
{
    const std::array<int, 7> INT_VECTOR_CONTENT = {43, -13, 90, -39, 0, 23, -75};
    const std::array<float, 8> FLOAT_VECTOR_CONTENT = {-2.124f, 23.2f - 82.32f, 2343.3f, 1.04f, 0.f, 321.3f, -9.f};
    const std::array<double, 10> DOUBLE_VECTOR_CONTENT = {3.14, -1.234, -0.234, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 3224.3423};
    Vector<int> initVectorInt(INT_VECTOR_CONTENT);
    Vector<float> initVectorFloat(FLOAT_VECTOR_CONTENT);
    Vector<double> initVectorDouble(DOUBLE_VECTOR_CONTENT);
    EXPECT_EQ(initVectorInt.Size(), INT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorFloat.Size(), FLOAT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorDouble.Size(), DOUBLE_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorInt.Dimension(), INT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorFloat.Dimension(), FLOAT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorDouble.Dimension(), DOUBLE_VECTOR_CONTENT.size());
    int i = 0;
    for (const auto &content : INT_VECTOR_CONTENT)
        EXPECT_EQ(initVectorInt[i++], content);
    i = 0;
    for (const auto &content : FLOAT_VECTOR_CONTENT)
        EXPECT_FLOAT_EQ(initVectorFloat[i++], content);
    i = 0;
    for (const auto &content : DOUBLE_VECTOR_CONTENT)
        EXPECT_DOUBLE_EQ(initVectorDouble[i++], content);
}

TEST(Vector, ZeroVector)
{
    const int VECTOR_LENGHTS[] = {0, 2, 4, 8, 16, 32, 64, 128, 256};
    for (int i = 0; i < sizeof(VECTOR_LENGHTS) / sizeof(VECTOR_LENGHTS[0]); i++)
    {
        auto zeroVector = Vector<int>::ZeroVector(VECTOR_LENGHTS[i]);
        EXPECT_EQ(zeroVector.Size(), VECTOR_LENGHTS[i]);
        EXPECT_EQ(zeroVector.Dimension(), VECTOR_LENGHTS[i]);
        for (int j = 0; j < VECTOR_LENGHTS[j]; j++)
            EXPECT_EQ(zeroVector[j], 0);
    }
}