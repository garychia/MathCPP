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

TEST(Vector, VectorStdVectorConstructor)
{
    const std::vector<int> INT_VECTOR_CONTENT({43, -13, 90, -39, 0, 23, -75});
    const std::vector<float> FLOAT_VECTOR_CONTENT({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    const std::vector<double> DOUBLE_VECTOR_CONTENT({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
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

TEST(Vector, VectorCopyConstructor)
{
    const std::vector<int> INT_VECTOR_CONTENT({43, -13, 90, -39, 0, 23, -75});
    const std::vector<float> FLOAT_VECTOR_CONTENT({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    const std::vector<double> DOUBLE_VECTOR_CONTENT({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> initVectorInt(INT_VECTOR_CONTENT);
    Vector<float> initVectorFloat(FLOAT_VECTOR_CONTENT);
    Vector<double> initVectorDouble(DOUBLE_VECTOR_CONTENT);
    Vector<int> initVectorIntCopy(initVectorInt);
    Vector<float> initVectorFloatCopy(initVectorFloat);
    Vector<double> initVectorDoubleCopy(initVectorDouble);
    EXPECT_EQ(initVectorInt.Size(), initVectorIntCopy.Size());
    EXPECT_EQ(initVectorFloat.Size(), initVectorFloatCopy.Size());
    EXPECT_EQ(initVectorDouble.Size(), initVectorDoubleCopy.Size());
    EXPECT_EQ(initVectorInt.Dimension(), initVectorIntCopy.Size());
    EXPECT_EQ(initVectorFloat.Dimension(), initVectorFloatCopy.Size());
    EXPECT_EQ(initVectorDouble.Dimension(), initVectorDoubleCopy.Size());
    for (int i = 0; i < initVectorInt.Size(); i++)
        EXPECT_EQ(initVectorInt[i], initVectorIntCopy[i]);
    for (int i = 0; i < initVectorFloat.Size(); i++)
        EXPECT_FLOAT_EQ(initVectorFloat[i], initVectorFloatCopy[i]);
    for (int i = 0; i < initVectorDouble.Size(); i++)
        EXPECT_DOUBLE_EQ(initVectorDouble[i], initVectorDoubleCopy[i]);
    // Copy Constructor with different types.
    // Int to Float.
    Vector<float> initVectorIntToFloatCopy(initVectorInt);
    EXPECT_EQ(initVectorIntToFloatCopy.Size(), initVectorInt.Size());
    for (int i = 0; i < initVectorIntToFloatCopy.Size(); i++)
        EXPECT_FLOAT_EQ(initVectorIntToFloatCopy[i], initVectorInt[i]);
    // Float to Double.
    Vector<float> initVectorFloatToDoubleCopy(initVectorFloat);
    EXPECT_EQ(initVectorFloatToDoubleCopy.Size(), initVectorFloat.Size());
    for (int i = 0; i < initVectorFloatToDoubleCopy.Size(); i++)
        EXPECT_DOUBLE_EQ(initVectorFloatToDoubleCopy[i], initVectorFloat[i]);
}

TEST(Vector, VectorMoveConstructor)
{
    const std::vector<int> INT_VECTOR_CONTENT({43, -13, 90, -39, 0, 23, -75, 23, 35, -93, 75, 46});
    const std::vector<float> FLOAT_VECTOR_CONTENT({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    const std::vector<double> DOUBLE_VECTOR_CONTENT({3.14, -1.24, -0.5576, -94.3, 0.0, -3.1343, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> initVectorInt(INT_VECTOR_CONTENT);
    Vector<float> initVectorFloat(FLOAT_VECTOR_CONTENT);
    Vector<double> initVectorDouble(DOUBLE_VECTOR_CONTENT);
    Vector<int> initVectorIntCopy = std::move(initVectorInt);
    Vector<float> initVectorFloatCopy = std::move(initVectorFloat);
    Vector<double> initVectorDoubleCopy = std::move(initVectorDouble);
    EXPECT_EQ(initVectorInt.Size(), ZERO);
    EXPECT_EQ(initVectorFloat.Size(), ZERO);
    EXPECT_EQ(initVectorDouble.Size(), ZERO);
    EXPECT_EQ(initVectorInt.Dimension(), ZERO);
    EXPECT_EQ(initVectorFloat.Dimension(), ZERO);
    EXPECT_EQ(initVectorDouble.Dimension(), ZERO);
    EXPECT_EQ(initVectorIntCopy.Size(), INT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorFloatCopy.Size(), FLOAT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorDoubleCopy.Size(), DOUBLE_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorIntCopy.Dimension(), INT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorFloatCopy.Dimension(), FLOAT_VECTOR_CONTENT.size());
    EXPECT_EQ(initVectorDoubleCopy.Dimension(), DOUBLE_VECTOR_CONTENT.size());
    for (int i = 0; i < initVectorInt.Size(); i++)
        EXPECT_EQ(initVectorInt[i], initVectorIntCopy[i]);
    for (int i = 0; i < initVectorFloat.Size(); i++)
        EXPECT_FLOAT_EQ(initVectorFloat[i], initVectorFloatCopy[i]);
    for (int i = 0; i < initVectorDouble.Size(); i++)
        EXPECT_DOUBLE_EQ(initVectorDouble[i], initVectorDoubleCopy[i]);
}

TEST(Vector, VectorCopyAssignment)
{
    const std::vector<int> INT_VECTOR_CONTENT({43, -13, 90, -39, 0, 23, -75});
    const std::vector<float> FLOAT_VECTOR_CONTENT({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    const std::vector<double> DOUBLE_VECTOR_CONTENT({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> initVectorInt(INT_VECTOR_CONTENT);
    Vector<float> initVectorFloat(FLOAT_VECTOR_CONTENT);
    Vector<double> initVectorDouble(DOUBLE_VECTOR_CONTENT);
    Vector<int> initVectorIntCopy = initVectorInt;
    Vector<float> initVectorFloatCopy = initVectorFloat;
    Vector<double> initVectorDoubleCopy = initVectorDouble;
    EXPECT_EQ(initVectorInt.Size(), initVectorIntCopy.Size());
    EXPECT_EQ(initVectorFloat.Size(), initVectorFloatCopy.Size());
    EXPECT_EQ(initVectorDouble.Size(), initVectorDoubleCopy.Size());
    EXPECT_EQ(initVectorInt.Dimension(), initVectorIntCopy.Size());
    EXPECT_EQ(initVectorFloat.Dimension(), initVectorFloatCopy.Size());
    EXPECT_EQ(initVectorDouble.Dimension(), initVectorDoubleCopy.Size());
    for (int i = 0; i < initVectorInt.Size(); i++)
        EXPECT_EQ(initVectorInt[i], initVectorIntCopy[i]);
    for (int i = 0; i < initVectorFloat.Size(); i++)
        EXPECT_FLOAT_EQ(initVectorFloat[i], initVectorFloatCopy[i]);
    for (int i = 0; i < initVectorDouble.Size(); i++)
        EXPECT_DOUBLE_EQ(initVectorDouble[i], initVectorDoubleCopy[i]);
    // Copy Assignment with different types.
    // Int to Float.
    initVectorFloatCopy = initVectorInt;
    EXPECT_EQ(initVectorFloatCopy.Size(), initVectorInt.Size());
    for (int i = 0; i < initVectorFloatCopy.Size(); i++)
        EXPECT_FLOAT_EQ(initVectorFloatCopy[i], initVectorInt[i]);
    // Float to Double.
    initVectorDoubleCopy = initVectorFloat;
    EXPECT_EQ(initVectorDoubleCopy.Size(), initVectorFloat.Size());
    for (int i = 0; i < initVectorDoubleCopy.Size(); i++)
        EXPECT_DOUBLE_EQ(initVectorDoubleCopy[i], initVectorFloat[i]);
}

TEST(Vector, VectorElementAccess)
{
    std::vector<int> elements;
    for (int i = -100; i < 101; i++)
        elements.push_back(i);
    Vector<int> t(elements);
    for (int i = 0; i < elements.size(); i++)
        EXPECT_EQ(elements[i], t[i]);
    EXPECT_THROW(
        {
            try
            {
                t[elements.size()];
            }
            catch (const Exceptions::IndexOutOfBound &e)
            {
                std::stringstream ss;
                ss << "Index Out of Bound: " << elements.size() << "\n";
                ss << "Vector: Index must be less than the dimension.";
                EXPECT_TRUE(ss.str() == e.what());
                throw e;
            }
        },
        Exceptions::IndexOutOfBound);
}

TEST(Vector, VectorDimension)
{
    Vector<int> t1;
    Vector<float> t2({1.f, 2.f, 3.f, 4.f});
    Vector<double> t3({32.3, 42.42, 53.75});
    EXPECT_EQ(t1.Dimension(), 0);
    EXPECT_EQ(t2.Dimension(), 4);
    EXPECT_EQ(t3.Dimension(), 3);
}

TEST(Vector, VectorLength)
{
    Vector<int> t1 = Vector<int>::ZeroVector(10);
    EXPECT_FLOAT_EQ(t1.Length<float>(), 0.f);
    Vector<float> t2({ 3.f, 4.f});
    EXPECT_FLOAT_EQ(t2.Length<float>(), 5.f);
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