#include <gtest/gtest.h>
#include <sstream>

#include "Tuple.hpp"

#define ZERO 0

using namespace DataStructures;

TEST(Tuple, TupleEmptyConstructor)
{
    Tuple<int> empty;
    EXPECT_EQ(empty.Size(), ZERO);
}

TEST(Tuple, TupleFillConstructor)
{
    const int TUPLE_LENGHTS[] = {0, 2, 4, 8, 16, 32, 64, 128, 256};
    for (int i = 0; i < sizeof(TUPLE_LENGHTS) / sizeof(TUPLE_LENGHTS[ZERO]); i++)
    {
        // Fill the vector with its size (dimension).
        Tuple<int> v(TUPLE_LENGHTS[i], TUPLE_LENGHTS[i]);
        EXPECT_EQ(v.Size(), TUPLE_LENGHTS[i]);
        for (int j = 0; j < TUPLE_LENGHTS[j]; j++)
            EXPECT_EQ(v[j], TUPLE_LENGHTS[i]);
    }
}

TEST(Tuple, TupleInitializerListConstructor)
{
    const auto INT_TUPLE_CONTENT = {43, -13, 90, -39, 0, 23, -75};
    const auto FLOAT_TUPLE_CONTENT = {-2.124f, 23.2f - 82.32f, 2343.3f, 1.04f, 0.f, 321.3f, -9.f};
    const auto DOUBLE_TUPLE_CONTENT = {3.14, -1.234, -0.234, -94.3, 0.0, 23.0, -7.5, 0.85};
    Tuple<int> initTupleInt(INT_TUPLE_CONTENT);
    Tuple<float> initTupleFloat(FLOAT_TUPLE_CONTENT);
    Tuple<double> initTupleDouble(DOUBLE_TUPLE_CONTENT);
    EXPECT_EQ(initTupleInt.Size(), INT_TUPLE_CONTENT.size());
    EXPECT_EQ(initTupleFloat.Size(), FLOAT_TUPLE_CONTENT.size());
    EXPECT_EQ(initTupleDouble.Size(), DOUBLE_TUPLE_CONTENT.size());
    int i = 0;
    for (const auto &content : INT_TUPLE_CONTENT)
        EXPECT_EQ(initTupleInt[i++], content);
    i = 0;
    for (const auto &content : FLOAT_TUPLE_CONTENT)
        EXPECT_FLOAT_EQ(initTupleFloat[i++], content);
    i = 0;
    for (const auto &content : DOUBLE_TUPLE_CONTENT)
        EXPECT_DOUBLE_EQ(initTupleDouble[i++], content);
}

TEST(Tuple, TupleArrayConstructor)
{
    const std::array<int, 7> INT_TUPLE_CONTENT = {43, -13, 90, -39, 0, 23, -75};
    const std::array<float, 8> FLOAT_TUPLE_CONTENT = {-2.124f, 23.2f, -82.32f, 2343.3f, 1.04f, 0.f, 321.3f, -9.f};
    const std::array<double, 10> DOUBLE_TUPLE_CONTENT = {3.14, -1.234, -0.234, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 3224.3423};
    Tuple<int> initTupleInt(INT_TUPLE_CONTENT);
    Tuple<float> initTupleFloat(FLOAT_TUPLE_CONTENT);
    Tuple<double> initTupleDouble(DOUBLE_TUPLE_CONTENT);
    EXPECT_EQ(initTupleInt.Size(), INT_TUPLE_CONTENT.size());
    EXPECT_EQ(initTupleFloat.Size(), FLOAT_TUPLE_CONTENT.size());
    EXPECT_EQ(initTupleDouble.Size(), DOUBLE_TUPLE_CONTENT.size());
    int i = 0;
    for (const auto &content : INT_TUPLE_CONTENT)
        EXPECT_EQ(initTupleInt[i++], content);
    i = 0;
    for (const auto &content : FLOAT_TUPLE_CONTENT)
        EXPECT_FLOAT_EQ(initTupleFloat[i++], content);
    i = 0;
    for (const auto &content : DOUBLE_TUPLE_CONTENT)
        EXPECT_DOUBLE_EQ(initTupleDouble[i++], content);
}

TEST(Tuple, TupleStdVectorConstructor)
{
    const std::vector<int> INT_TUPLE_CONTENT({43, -13, 90, -39, 0, 23, -75});
    const std::vector<float> FLOAT_TUPLE_CONTENT({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    const std::vector<double> DOUBLE_TUPLE_CONTENT({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Tuple<int> initTupleInt(INT_TUPLE_CONTENT);
    Tuple<float> initTupleFloat(FLOAT_TUPLE_CONTENT);
    Tuple<double> initTupleDouble(DOUBLE_TUPLE_CONTENT);
    EXPECT_EQ(initTupleInt.Size(), INT_TUPLE_CONTENT.size());
    EXPECT_EQ(initTupleFloat.Size(), FLOAT_TUPLE_CONTENT.size());
    EXPECT_EQ(initTupleDouble.Size(), DOUBLE_TUPLE_CONTENT.size());
    int i = 0;
    for (const auto &content : INT_TUPLE_CONTENT)
        EXPECT_EQ(initTupleInt[i++], content);
    i = 0;
    for (const auto &content : FLOAT_TUPLE_CONTENT)
        EXPECT_FLOAT_EQ(initTupleFloat[i++], content);
    i = 0;
    for (const auto &content : DOUBLE_TUPLE_CONTENT)
        EXPECT_DOUBLE_EQ(initTupleDouble[i++], content);
}

TEST(Tuple, TupleCopyConstructor)
{
    const std::vector<int> INT_TUPLE_CONTENT({43, -13, 90, -39, 0, 23, -75});
    const std::vector<float> FLOAT_TUPLE_CONTENT({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    const std::vector<double> DOUBLE_TUPLE_CONTENT({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Tuple<int> initTupleInt(INT_TUPLE_CONTENT);
    Tuple<float> initTupleFloat(FLOAT_TUPLE_CONTENT);
    Tuple<double> initTupleDouble(DOUBLE_TUPLE_CONTENT);
    Tuple<int> initTupleIntCopy(initTupleInt);
    Tuple<float> initTupleFloatCopy(initTupleFloat);
    Tuple<double> initTupleDoubleCopy(initTupleDouble);
    EXPECT_EQ(initTupleInt.Size(), initTupleIntCopy.Size());
    EXPECT_EQ(initTupleFloat.Size(), initTupleFloatCopy.Size());
    EXPECT_EQ(initTupleDouble.Size(), initTupleDoubleCopy.Size());
    for (int i = 0; i < initTupleInt.Size(); i++)
        EXPECT_EQ(initTupleInt[i], initTupleIntCopy[i]);
    for (int i = 0; i < initTupleFloat.Size(); i++)
        EXPECT_FLOAT_EQ(initTupleFloat[i], initTupleFloatCopy[i]);
    for (int i = 0; i < initTupleDouble.Size(); i++)
        EXPECT_DOUBLE_EQ(initTupleDouble[i], initTupleDoubleCopy[i]);
    // Copy Constructor with different types.
    // Int to Float.
    Tuple<float> initTupleIntToFloatCopy(initTupleInt);
    EXPECT_EQ(initTupleIntToFloatCopy.Size(), initTupleInt.Size());
    for (int i = 0; i < initTupleIntToFloatCopy.Size(); i++)
        EXPECT_FLOAT_EQ(initTupleIntToFloatCopy[i], initTupleInt[i]);
    // Float to Double.
    Tuple<float> initTupleFloatToDoubleCopy(initTupleFloat);
    EXPECT_EQ(initTupleFloatToDoubleCopy.Size(), initTupleFloat.Size());
    for (int i = 0; i < initTupleFloatToDoubleCopy.Size(); i++)
        EXPECT_DOUBLE_EQ(initTupleFloatToDoubleCopy[i], initTupleFloat[i]);
}

TEST(Tuple, TupleMoveConstructor)
{
    const std::vector<int> INT_TUPLE_CONTENT({43, -13, 90, -39, 0, 23, -75, 23, 35, -93, 75, 46});
    const std::vector<float> FLOAT_TUPLE_CONTENT({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    const std::vector<double> DOUBLE_TUPLE_CONTENT({3.14, -1.24, -0.5576, -94.3, 0.0, -3.1343, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Tuple<int> initTupleInt(INT_TUPLE_CONTENT);
    Tuple<float> initTupleFloat(FLOAT_TUPLE_CONTENT);
    Tuple<double> initTupleDouble(DOUBLE_TUPLE_CONTENT);
    Tuple<int> initTupleIntCopy = std::move(initTupleInt);
    Tuple<float> initTupleFloatCopy = std::move(initTupleFloat);
    Tuple<double> initTupleDoubleCopy = std::move(initTupleDouble);
    EXPECT_EQ(initTupleInt.Size(), ZERO);
    EXPECT_EQ(initTupleFloat.Size(), ZERO);
    EXPECT_EQ(initTupleDouble.Size(), ZERO);
    EXPECT_EQ(initTupleIntCopy.Size(), INT_TUPLE_CONTENT.size());
    EXPECT_EQ(initTupleFloatCopy.Size(), FLOAT_TUPLE_CONTENT.size());
    EXPECT_EQ(initTupleDoubleCopy.Size(), DOUBLE_TUPLE_CONTENT.size());
    for (int i = 0; i < initTupleInt.Size(); i++)
        EXPECT_EQ(initTupleInt[i], initTupleIntCopy[i]);
    for (int i = 0; i < initTupleFloat.Size(); i++)
        EXPECT_FLOAT_EQ(initTupleFloat[i], initTupleFloatCopy[i]);
    for (int i = 0; i < initTupleDouble.Size(); i++)
        EXPECT_DOUBLE_EQ(initTupleDouble[i], initTupleDoubleCopy[i]);
}

TEST(Tuple, TupleCopyAssignmentConstructor)
{
    const std::vector<int> INT_TUPLE_CONTENT({43, -13, 90, -39, 0, 23, -75});
    const std::vector<float> FLOAT_TUPLE_CONTENT({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    const std::vector<double> DOUBLE_TUPLE_CONTENT({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Tuple<int> initTupleInt(INT_TUPLE_CONTENT);
    Tuple<float> initTupleFloat(FLOAT_TUPLE_CONTENT);
    Tuple<double> initTupleDouble(DOUBLE_TUPLE_CONTENT);
    Tuple<int> initTupleIntCopy = initTupleInt;
    Tuple<float> initTupleFloatCopy = initTupleFloat;
    Tuple<double> initTupleDoubleCopy = initTupleDouble;
    EXPECT_EQ(initTupleInt.Size(), initTupleIntCopy.Size());
    EXPECT_EQ(initTupleFloat.Size(), initTupleFloatCopy.Size());
    EXPECT_EQ(initTupleDouble.Size(), initTupleDoubleCopy.Size());
    for (int i = 0; i < initTupleInt.Size(); i++)
        EXPECT_EQ(initTupleInt[i], initTupleIntCopy[i]);
    for (int i = 0; i < initTupleFloat.Size(); i++)
        EXPECT_FLOAT_EQ(initTupleFloat[i], initTupleFloatCopy[i]);
    for (int i = 0; i < initTupleDouble.Size(); i++)
        EXPECT_DOUBLE_EQ(initTupleDouble[i], initTupleDoubleCopy[i]);
    // Copy Assignment with different types.
    // Int to Float.
    initTupleFloatCopy = initTupleInt;
    EXPECT_EQ(initTupleFloatCopy.Size(), initTupleInt.Size());
    for (int i = 0; i < initTupleFloatCopy.Size(); i++)
        EXPECT_FLOAT_EQ(initTupleFloatCopy[i], initTupleInt[i]);
    // Float to Double.
    initTupleDoubleCopy = initTupleFloat;
    EXPECT_EQ(initTupleDoubleCopy.Size(), initTupleFloat.Size());
    for (int i = 0; i < initTupleDoubleCopy.Size(); i++)
        EXPECT_DOUBLE_EQ(initTupleDoubleCopy[i], initTupleFloat[i]);
}

TEST(Tuple, TupleElementAccess)
{
    std::vector<int> elements;
    for (int i = -100; i < 101; i++)
        elements.push_back(i);
    Tuple<int> t(elements);
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
                ss << "Tuple: Index must be non-negative and less than the number of elements.";
                EXPECT_TRUE(ss.str() == e.what());
                throw e;
            }
        },
        Exceptions::IndexOutOfBound);
}