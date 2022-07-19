#include <gtest/gtest.h>
#include <cmath>

#include "Vector.hpp"

#define ZERO 0

using namespace DataStructures;

TEST(Vector, EmptyConstructor)
{
    Vector<int> empty;
    EXPECT_EQ(empty.Size(), ZERO);
    EXPECT_EQ(empty.Dimension(), ZERO);
}

TEST(Vector, FillConstructor)
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

TEST(Vector, InitializerListConstructor)
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

TEST(Vector, ArrayConstructor)
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

TEST(Vector, StdVectorConstructor)
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

TEST(Vector, CopyConstructor)
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

TEST(Vector, MoveConstructor)
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

TEST(Vector, CopyAssignment)
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

TEST(Vector, ElementAccess)
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

TEST(Vector, Dimension)
{
    Vector<int> v1;
    Vector<float> v2({1.f, 2.f, 3.f, 4.f});
    Vector<double> v3({32.3, 42.42, 53.75});
    EXPECT_EQ(v1.Dimension(), 0);
    EXPECT_EQ(v2.Dimension(), 4);
    EXPECT_EQ(v3.Dimension(), 3);
}

TEST(Vector, Length)
{
    Vector<int> v1 = Vector<int>::ZeroVector(10);
    EXPECT_FLOAT_EQ(v1.Length<float>(), 0.f);
    Vector<float> v2({3.f, 4.f});
    EXPECT_FLOAT_EQ(v2.Length<float>(), 5.f);
    Vector<float> v3({-3.f, 4.f});
    EXPECT_FLOAT_EQ(v3.Length<float>(), 5.f);
    Vector<float> t4({3.f, -4.f});
    EXPECT_FLOAT_EQ(t4.Length<float>(), 5.f);
    Vector<float> t5({-3.f, -4.f});
    EXPECT_FLOAT_EQ(t5.Length<float>(), 5.f);
    Vector<int> t0;
    EXPECT_THROW(
        try {
            t0.Length<float>();
        } catch (const Exceptions::EmptyVector &e) {
            std::stringstream ss;
            ss << "Vector: Length of an empty vector is undefined.";
            EXPECT_TRUE(e.what() == ss.str());
            throw e;
        },
        Exceptions::EmptyVector);
}

TEST(Vector, EuclideanNorm)
{
    Vector<int> v1 = Vector<int>::ZeroVector(10);
    EXPECT_FLOAT_EQ(v1.EuclideanNorm<float>(), 0.f);
    Vector<float> v2({3.f, 4.f});
    EXPECT_FLOAT_EQ(v2.EuclideanNorm<float>(), 5.f);
    Vector<float> v3({-3.f, 4.f});
    EXPECT_FLOAT_EQ(v3.EuclideanNorm<float>(), 5.f);
    Vector<float> t4({3.f, -4.f});
    EXPECT_FLOAT_EQ(t4.EuclideanNorm<float>(), 5.f);
    Vector<float> t5({-3.f, -4.f});
    EXPECT_FLOAT_EQ(t5.EuclideanNorm<float>(), 5.f);

    Vector<int> t0;
    EXPECT_THROW(
        try {
            t0.EuclideanNorm<float>();
        } catch (const Exceptions::EmptyVector &e) {
            std::stringstream ss;
            ss << "Vector: Euclidean norm of an empty vector is undefined.";
            EXPECT_TRUE(e.what() == ss.str());
            throw e;
        },
        Exceptions::EmptyVector);
}

template <class T>
void CheckVectorLpNorm(const Vector<T> &v, int p)
{
    if (v.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v.template LpNorm<double>(p);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Lp norm of an empty vector is undefined.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    double sum = 0;
    for (std::size_t i = 0; i < v.Size(); i++)
        sum += pow(v[i], p);
    EXPECT_NEAR(v.template LpNorm<double>(p), pow(sum, (double)1 / p), 0.0001);
}

TEST(Vector, LpNorm)
{
    Vector<float> v1({3.f, 4.f});
    CheckVectorLpNorm(v1, 2);
    Vector<int> v2({4, 5, 64, 2, 67, 32, 54});
    CheckVectorLpNorm(v2, 5);
    Vector<double> v3({23.345, 4.43, 5.345, 64.23, 32.56, 6.4537, 32.536, 54.4535, 2684.567});
    CheckVectorLpNorm(v3, 8);
    Vector<int> t0;
    CheckVectorLpNorm(t0, 32);
}

template <class T, class U>
void CheckVectorAddition(const Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Add(v2);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addition on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Add(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addtion on the given empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Size() % v2.Size() != 0)
    {
        EXPECT_THROW(
            try {
                v1.Add(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const auto result = v1.Add(v2);
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] + v2[i % v2.Size()], result[i]);
}

TEST(Vector, AddVectors)
{
    Vector<int> v1({43, -13});
    Vector<int> v2({96, -4, 99, 83, 48, -263, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> v0;
    CheckVectorAddition(v1, v2);
    CheckVectorAddition(v2, v1);
    CheckVectorAddition(v1, v3);
    CheckVectorAddition(v3, v1);
    CheckVectorAddition(v1, v4);
    CheckVectorAddition(v4, v1);
    CheckVectorAddition(v0, v4);
    CheckVectorAddition(v4, v0);
}

template <class T, class Scaler>
void CheckScalerAddition(const Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Add(s);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addition on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto result = v1.Add(s);
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] + s, result[i]);
}

TEST(Vector, AddScaler)
{
    Vector<int> v1({43, -13});
    Vector<int> v2({96, -4, 99, 83, 48, -263, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> v0;
    const int s1 = 32;
    const float s2 = 3.1415f;
    const double s3 = 56635.45245;
    CheckScalerAddition(v1, s1);
    CheckScalerAddition(v1, s2);
    CheckScalerAddition(v1, s3);
    CheckScalerAddition(v2, s1);
    CheckScalerAddition(v2, s2);
    CheckScalerAddition(v2, s3);
    CheckScalerAddition(v3, s1);
    CheckScalerAddition(v3, s2);
    CheckScalerAddition(v3, s3);
    CheckScalerAddition(v4, s1);
    CheckScalerAddition(v4, s2);
    CheckScalerAddition(v4, s3);
    CheckScalerAddition(v0, s3);
}

template <class T, class U>
void CheckOperatorPlusVector(const Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 + v2;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addition on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 + v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addtion on the given empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Size() % v2.Size() != 0)
    {
        EXPECT_THROW(
            try {
                v1 + v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const auto result = v1 + v2;
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] + v2[i % v2.Size()], result[i]);
}

TEST(Vector, OperatorPlusVector)
{
    Vector<int> v1({43, -13});
    Vector<int> v2({96, -4, 99, 83, 48, -263, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> v0;
    CheckOperatorPlusVector(v1, v2);
    CheckOperatorPlusVector(v2, v1);
    CheckOperatorPlusVector(v1, v3);
    CheckOperatorPlusVector(v3, v1);
    CheckOperatorPlusVector(v1, v4);
    CheckOperatorPlusVector(v4, v1);
    CheckOperatorPlusVector(v0, v1);
    CheckOperatorPlusVector(v1, v0);
}

template <class T, class Scaler>
void CheckOperatorPlusScaler(const Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 + s;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addition on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto result = v1 + s;
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] + s, result[i]);
}

TEST(Vector, OperatorPlusScaler)
{
    Vector<int> v1({43, -13});
    Vector<int> v2({96, -4, 99, 83, 48, -263, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> v0;
    const int s1 = 32;
    const float s2 = 3.1415f;
    const double s3 = 56635.45245;
    CheckOperatorPlusScaler(v1, s1);
    CheckOperatorPlusScaler(v1, s2);
    CheckOperatorPlusScaler(v1, s3);
    CheckOperatorPlusScaler(v2, s1);
    CheckOperatorPlusScaler(v2, s2);
    CheckOperatorPlusScaler(v2, s3);
    CheckOperatorPlusScaler(v3, s1);
    CheckOperatorPlusScaler(v3, s2);
    CheckOperatorPlusScaler(v3, s3);
    CheckOperatorPlusScaler(v4, s1);
    CheckOperatorPlusScaler(v4, s2);
    CheckOperatorPlusScaler(v4, s3);
    CheckOperatorPlusScaler(v0, s3);
}

template <class T, class U>
void CheckOperatorPlusAssignmentVector(Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 += v2;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addition on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 += v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addtion on the given empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Size() % v2.Size() != 0)
    {
        EXPECT_THROW(
            try {
                v1 += v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const auto result = v1 + v2;
    v1 += v2;
    for (std::size_t i = 0; i < result.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i], result[i]);
}

TEST(Vector, OperatorPlusAssignmentVector)
{
    Vector<int> v1({43, -13});
    Vector<int> v2({96, -4, 99, 83, 48, -263, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> v0;
    CheckOperatorPlusAssignmentVector(v1, v2);
    CheckOperatorPlusAssignmentVector(v2, v1);
    CheckOperatorPlusAssignmentVector(v1, v3);
    CheckOperatorPlusAssignmentVector(v3, v1);
    CheckOperatorPlusAssignmentVector(v1, v4);
    CheckOperatorPlusAssignmentVector(v4, v1);
    CheckOperatorPlusAssignmentVector(v0, v2);
    CheckOperatorPlusAssignmentVector(v2, v0);
}

template <class T, class Scaler>
void CheckOperatorPlusAssignmentScaler(Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 += s;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addition on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const Vector<T> result = v1 + s;
    v1 += s;
    for (std::size_t i = 0; i < result.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i], result[i]);
}

TEST(Vector, OperatorPlusAssignmentScaler)
{
    Vector<int> v1({43, -13});
    Vector<int> v2({96, -4, 99, 83, 48, -263, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> v0;
    const int s1 = 32;
    const float s2 = 3.1415f;
    const double s3 = 56635.45245;
    CheckOperatorPlusAssignmentScaler(v1, s1);
    CheckOperatorPlusAssignmentScaler(v1, s2);
    CheckOperatorPlusAssignmentScaler(v1, s3);
    CheckOperatorPlusAssignmentScaler(v2, s1);
    CheckOperatorPlusAssignmentScaler(v2, s2);
    CheckOperatorPlusAssignmentScaler(v2, s3);
    CheckOperatorPlusAssignmentScaler(v3, s1);
    CheckOperatorPlusAssignmentScaler(v3, s2);
    CheckOperatorPlusAssignmentScaler(v3, s3);
    CheckOperatorPlusAssignmentScaler(v4, s1);
    CheckOperatorPlusAssignmentScaler(v4, s2);
    CheckOperatorPlusAssignmentScaler(v4, s3);
    CheckOperatorPlusAssignmentScaler(v0, s3);
}

template <class T, class U>
void CheckVectorSubtraction(const Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Minus(v2);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Minus(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on the given empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Size() % v2.Size() != 0)
    {
        EXPECT_THROW(
            try {
                v1.Minus(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const auto result = v1.Minus(v2);
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] - v2[i % v2.Size()], result[i]);
}

TEST(Vector, Minus)
{
    Vector<int> v1({64, -13});
    Vector<int> v2({96, -4, 234, 83, 64, -23, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<int> v0;
    CheckVectorSubtraction(v1, v2);
    CheckVectorSubtraction(v2, v1);
    CheckVectorSubtraction(v1, v3);
    CheckVectorSubtraction(v3, v1);
    CheckVectorSubtraction(v1, v4);
    CheckVectorSubtraction(v4, v1);
    CheckVectorSubtraction(v0, v1);
    CheckVectorSubtraction(v1, v0);
    CheckVectorSubtraction(v0, v0);
}

template <class T, class Scaler>
void CheckScalerSubtraction(const Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Minus(s);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto result = v1.Minus(s);
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] - s, result[i]);
}

TEST(Vector, MinusScaler)
{
    Vector<int> v1({55, -19});
    Vector<int> v2({96, -4, 34, 83, 48, -286, 34, 325});
    Vector<float> v3({-2.1454f, 243.2f, -582.32f, 874.3f, 165.04f, 10.f, 332.3f, 0.f, 23.f});
    Vector<double> v4({23.435, -1.24454, -0.55676, -964.3, 0.0, 23.0, -7.45455, 0.4485, 1.2323, 2.3423});
    Vector<int> v0;
    const int s1 = 322342;
    const float s2 = 25873.1415f;
    const double s3 = 543.5644345;
    CheckScalerSubtraction(v1, s1);
    CheckScalerSubtraction(v1, s2);
    CheckScalerSubtraction(v1, s3);
    CheckScalerSubtraction(v2, s1);
    CheckScalerSubtraction(v2, s2);
    CheckScalerSubtraction(v2, s3);
    CheckScalerSubtraction(v3, s1);
    CheckScalerSubtraction(v3, s2);
    CheckScalerSubtraction(v3, s3);
    CheckScalerSubtraction(v4, s1);
    CheckScalerSubtraction(v4, s2);
    CheckScalerSubtraction(v4, s3);
    CheckScalerSubtraction(v0, s1);
}

template <class T, class U>
void CheckOperatorMinusVector(const Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 - v2;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 - v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on the given empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Size() % v2.Size() != 0)
    {
        EXPECT_THROW(
            try {
                v1 - v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const auto result = v1 - v2;
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] - v2[i % v2.Size()], result[i]);
}

TEST(Vector, OperatorMinusVector)
{
    Vector<int> v1({64, -13});
    Vector<int> v2({96, -4, 234, 83, 64, -23, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<int> v0;
    CheckOperatorMinusVector(v1, v2);
    CheckOperatorMinusVector(v2, v1);
    CheckOperatorMinusVector(v1, v3);
    CheckOperatorMinusVector(v3, v1);
    CheckOperatorMinusVector(v1, v4);
    CheckOperatorMinusVector(v4, v1);
    CheckOperatorMinusVector(v0, v1);
    CheckOperatorMinusVector(v1, v0);
    CheckOperatorMinusVector(v0, v0);
}

template <class T, class Scaler>
void CheckOperatorMinusScaler(const Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 - s;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto result = v1 - s;
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] - s, result[i]);
}

TEST(Vector, OperatorMinusScaler)
{
    Vector<int> v1({-4542, 34856});
    Vector<int> v2({96, -234, 34534, 89063, 24189, -2856, 9056534, 805325});
    Vector<float> v3({-5636.1454f, 243.2f, -582.32f, 874.3f, 23.234f, 1540.f, 332.3f, 6800450.f, 23.34532f});
    Vector<double> v4({23.435, -1.24454, -923.55676, -964.3, 0.0, 23.0324, -7.45455, 0.4485, 1.2323, 2.3423});
    Vector<int> v0;
    const int s1 = 322342;
    const float s2 = 25873.631415f;
    const double s3 = 543.885644345;
    CheckOperatorMinusScaler(v1, s1);
    CheckOperatorMinusScaler(v1, s2);
    CheckOperatorMinusScaler(v1, s3);
    CheckOperatorMinusScaler(v2, s1);
    CheckOperatorMinusScaler(v2, s2);
    CheckOperatorMinusScaler(v2, s3);
    CheckOperatorMinusScaler(v3, s1);
    CheckOperatorMinusScaler(v3, s2);
    CheckOperatorMinusScaler(v3, s3);
    CheckOperatorMinusScaler(v4, s1);
    CheckOperatorMinusScaler(v4, s2);
    CheckOperatorMinusScaler(v4, s3);
    CheckOperatorMinusScaler(v0, s3);
}

template <class T, class U>
void CheckOperatorMinusAssignmentVector(Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 -= v2;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 -= v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on the given empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Size() % v2.Size() != 0)
    {
        EXPECT_THROW(
            try {
                v1 -= v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const auto result = v1 - v2;
    v1 -= v2;
    for (std::size_t i = 0; i < result.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i], result[i]);
}

TEST(Vector, OperatorMinusAssignmentVector)
{
    Vector<int> v1({64, -13});
    Vector<int> v2({96, -4, 234, 83, 64, -23, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<int> v0;
    CheckOperatorMinusAssignmentVector(v1, v2);
    CheckOperatorMinusAssignmentVector(v2, v1);
    CheckOperatorMinusAssignmentVector(v1, v3);
    CheckOperatorMinusAssignmentVector(v3, v1);
    CheckOperatorMinusAssignmentVector(v1, v4);
    CheckOperatorMinusAssignmentVector(v4, v1);
    CheckOperatorMinusAssignmentVector(v0, v1);
    CheckOperatorMinusAssignmentVector(v1, v0);
    CheckOperatorMinusAssignmentVector(v0, v0);
}

template <class T, class Scaler>
void CheckOperatorMinusAssignmentScaler(Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 -= s;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const Vector<T> result = v1 - s;
    v1 -= s;
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i], result[i]);
}

TEST(Vector, OperatorMinusAssginmentScaler)
{
    Vector<int> v1({-4542, 34856});
    Vector<int> v2({96, -234, 34534, 89063, 24189, -2856, 9056534, 805325});
    Vector<float> v3({-5636.1454f, 243.2f, -582.32f, 874.3f, 23.234f, 1540.f, 332.3f, 6800450.f, 23.34532f});
    Vector<double> v4({23.435, -1.24454, -923.55676, -964.3, 0.0, 23.0324, -7.45455, 0.4485, 1.2323, 2.3423});
    Vector<int> v0;
    const int s1 = 322342;
    const float s2 = 25873.631415f;
    const double s3 = 543.885644345;
    CheckOperatorMinusAssignmentScaler(v1, s1);
    CheckOperatorMinusAssignmentScaler(v1, s2);
    CheckOperatorMinusAssignmentScaler(v1, s3);
    CheckOperatorMinusAssignmentScaler(v2, s1);
    CheckOperatorMinusAssignmentScaler(v2, s2);
    CheckOperatorMinusAssignmentScaler(v2, s3);
    CheckOperatorMinusAssignmentScaler(v3, s1);
    CheckOperatorMinusAssignmentScaler(v3, s2);
    CheckOperatorMinusAssignmentScaler(v3, s3);
    CheckOperatorMinusAssignmentScaler(v4, s1);
    CheckOperatorMinusAssignmentScaler(v4, s2);
    CheckOperatorMinusAssignmentScaler(v4, s3);
    CheckOperatorMinusAssignmentScaler(v0, s3);
}

template <class T, class Scaler>
void CheckScaleScaler(const Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Scale(s);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform scaling on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto result = v1.Scale(s);
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] * s, result[i]);
}

TEST(Vector, ScaleScaler)
{
    Vector<int> v1({-4542, 34856, 7435, 438, -2594});
    Vector<int> v2({96, -234, 34534, 89063, 24189, -2856, 6, 805325, 934});
    Vector<float> v3({-5636.1454f, 243.2f, -582.32f, 874.3f, 23.234f, 1540.f, 332.3f, 6800450.f, 23.34532f});
    Vector<double> v4({23.435, -1.24454, -923.55676, -964.3, 0.0, 23.0324, -7.45455, 0.4485, 1.2323, 2.3423});
    Vector<int> v0;
    const int s1 = -12;
    const float s2 = 25873.631415f;
    const double s3 = 543.885644345;
    CheckScaleScaler(v1, s1);
    CheckScaleScaler(v1, s2);
    CheckScaleScaler(v1, s3);
    CheckScaleScaler(v2, s1);
    CheckScaleScaler(v2, s2);
    CheckScaleScaler(v2, s3);
    CheckScaleScaler(v3, s1);
    CheckScaleScaler(v3, s2);
    CheckScaleScaler(v3, s3);
    CheckScaleScaler(v4, s1);
    CheckScaleScaler(v4, s2);
    CheckScaleScaler(v4, s3);
    CheckScaleScaler(v0, s3);
}

template <class T, class U>
void CheckScaleVector(const Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Scale(v2);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform scaling on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Scale(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform scaling with an empty vector as the second operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Dimension() % v2.Dimension() != 0)
    {
        EXPECT_THROW(
            try {
                v1.Scale(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const auto result = v1.Scale(v2);
    for (std::size_t i = 0; i < result.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] * v2[i % v2.Dimension()], result[i]);
}

TEST(Vector, ScaleVector)
{
    Vector<int> v1({64, -13, 943});
    Vector<int> v2({269, -34, 43, 283, 364, -323, 734, 849});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<int> v0;
    CheckScaleVector(v1, v2);
    CheckScaleVector(v2, v1);
    CheckScaleVector(v1, v3);
    CheckScaleVector(v3, v1);
    CheckScaleVector(v1, v4);
    CheckScaleVector(v4, v1);
    CheckScaleVector(v0, v1);
    CheckScaleVector(v1, v0);
    CheckScaleVector(v0, v0);
}

template <class T, class Scaler>
void CheckOperatorMultiplyScaler(const Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 *s;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform scaling on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto result = v1 * s;
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] * s, result[i]);
}

TEST(Vector, OperatorMultiplyScaler)
{
    Vector<int> v1({-4542, 34856, 7435, 438, -2594});
    Vector<int> v2({96, -234, 34534, 89063, 24189, -2856, 6, 805325, 934});
    Vector<float> v3({-5636.1454f, 243.2f, -582.32f, 874.3f, 23.234f, 1540.f, 332.3f, 6800450.f, 23.34532f});
    Vector<double> v4({23.435, -1.24454, -923.55676, -964.3, 0.0, 23.0324, -7.45455, 0.4485, 1.2323, 2.3423});
    Vector<int> v0;
    const int s1 = -12;
    const float s2 = 525873.631415f;
    const double s3 = 454453.885644345;
    CheckOperatorMultiplyScaler(v1, s1);
    CheckOperatorMultiplyScaler(v1, s2);
    CheckOperatorMultiplyScaler(v1, s3);
    CheckOperatorMultiplyScaler(v2, s1);
    CheckOperatorMultiplyScaler(v2, s2);
    CheckOperatorMultiplyScaler(v2, s3);
    CheckOperatorMultiplyScaler(v3, s1);
    CheckOperatorMultiplyScaler(v3, s2);
    CheckOperatorMultiplyScaler(v3, s3);
    CheckOperatorMultiplyScaler(v4, s1);
    CheckOperatorMultiplyScaler(v4, s2);
    CheckOperatorMultiplyScaler(v4, s3);
    CheckOperatorMultiplyScaler(v0, s3);
}

template <class T, class U>
void CheckOperatorMultiplyVector(const Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1 *v2;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise multiplication on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1 *v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise multiplication with an empty vector as the second operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Dimension() % v2.Dimension() != 0)
    {
        EXPECT_THROW(
            try {
                v1 *v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Expect the dimension of the second operand is a factor of that "
                      "of the first operand when performing element-wise multiplication.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const auto result = v1 * v2;
    for (std::size_t i = 0; i < result.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] * v2[i % v2.Dimension()], result[i]);
}

TEST(Vector, OperatorMultiplyVector)
{
    Vector<int> v1({64, -13, 943});
    Vector<int> v2({269, -34, 43, 283, 364, -323, 734, 849});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<int> v0;
    CheckOperatorMultiplyVector(v1, v2);
    CheckOperatorMultiplyVector(v2, v1);
    CheckOperatorMultiplyVector(v1, v3);
    CheckOperatorMultiplyVector(v3, v1);
    CheckOperatorMultiplyVector(v1, v4);
    CheckOperatorMultiplyVector(v4, v1);
    CheckOperatorMultiplyVector(v0, v1);
    CheckOperatorMultiplyVector(v1, v0);
    CheckOperatorMultiplyVector(v0, v0);
}

template <class T, class Scaler>
void CheckOperatorMultiplyAssignmentScaler(Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 *= s;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform scaling on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto v1Copy = v1;
    v1 *= s;
    for (std::size_t i = 0; i < v1.Dimension(); i++)
        EXPECT_DOUBLE_EQ(v1[i], T(v1Copy[i] * s));
}

TEST(Vector, OperatorMultiplyAssignmentScaler)
{
    Vector<int> v1({-34, 243, -7435, 4554, -4});
    Vector<int> v2({96, -234, 12, -43, 56, -89, 6, 64, 934});
    Vector<float> v3({-103.1454f, 13.2f, -75.32f, 74.3f, -23.234f, 67.f, 53.3f, 434.f, 23.565});
    Vector<double> v4({23.435, -1.24454, -421.55676, -403.3, 324.0, 23.0324, -7.45455, 0.4485, 71.756, 42.3423});
    Vector<int> v0;
    const int s1 = -234;
    const float s2 = 34.4378;
    const double s3 = 905.2345;
    CheckOperatorMultiplyAssignmentScaler(v1, s1);
    CheckOperatorMultiplyAssignmentScaler(v1, s2);
    CheckOperatorMultiplyAssignmentScaler(v1, s3);
    CheckOperatorMultiplyAssignmentScaler(v2, s1);
    CheckOperatorMultiplyAssignmentScaler(v2, s2);
    CheckOperatorMultiplyAssignmentScaler(v2, s3);
    CheckOperatorMultiplyAssignmentScaler(v3, s1);
    CheckOperatorMultiplyAssignmentScaler(v3, s2);
    CheckOperatorMultiplyAssignmentScaler(v3, s3);
    CheckOperatorMultiplyAssignmentScaler(v4, s1);
    CheckOperatorMultiplyAssignmentScaler(v4, s2);
    CheckOperatorMultiplyAssignmentScaler(v4, s3);
    CheckOperatorMultiplyAssignmentScaler(v0, s3);
}

template <class T, class U>
void CheckOperatorMultiplyAssignmentVector(Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 *= v2;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise multiplication on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 *= v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise multiplication with an empty vector as the second operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Size() % v2.Size() != 0)
    {
        EXPECT_THROW(
            try {
                v1 *= v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Expect the dimension of the second operand is a factor of that "
                      "of the first operand when performing element-wise multiplication.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const Vector<T> result = v1 * v2;
    v1 *= v2;
    for (std::size_t i = 0; i < result.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i], result[i]);
}

TEST(Vector, OperatorMultiplyAssignmentVector)
{
    Vector<int> v1({64, -13});
    Vector<int> v2({96, -4, 234, 83, 64, -23, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<int> v0;
    CheckOperatorMultiplyAssignmentVector(v1, v2);
    CheckOperatorMultiplyAssignmentVector(v2, v1);
    CheckOperatorMultiplyAssignmentVector(v1, v3);
    CheckOperatorMultiplyAssignmentVector(v3, v1);
    CheckOperatorMultiplyAssignmentVector(v1, v4);
    CheckOperatorMultiplyAssignmentVector(v4, v1);
    CheckOperatorMultiplyAssignmentVector(v0, v1);
    CheckOperatorMultiplyAssignmentVector(v1, v0);
    CheckOperatorMultiplyAssignmentVector(v0, v0);
}

template <class T, class Scaler>
void CheckDivideScaler(const Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Divide(s);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (s == 0)
    {
        EXPECT_THROW(
            try {
                v1.Divide(s);
            } catch (const Exceptions::DividedByZero &e) {
                std::stringstream ss;
                ss << "Division by zero occurred.\n";
                ss << "Vector: Cannot perform element-wise division as the second operand is 0.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::DividedByZero);
        return;
    }
    const auto result = v1.Divide(s);
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] / s, result[i]);
}

TEST(Vector, DivideScaler)
{
    Vector<int> v1({-4542, 34856, 7435, 438, -2594});
    Vector<int> v2({96, -234, 34534, 89063, 24189, -2856, 6, 805325, 934});
    Vector<float> v3({-5636.1454f, 243.2f, -582.32f, 874.3f, 23.234f, 1540.f, 332.3f, 6800450.f, 23.34532f});
    Vector<double> v4({23.435, -1.24454, -923.55676, -964.3, 0.0, 23.0324, -7.45455, 0.4485, 1.2323, 2.3423});
    Vector<int> v0;
    const int s1 = -12;
    const float s2 = 525873.631415f;
    const double s3 = 454453.885644345;
    const int s0 = 0;
    CheckDivideScaler(v1, s1);
    CheckDivideScaler(v1, s2);
    CheckDivideScaler(v1, s3);
    CheckDivideScaler(v1, s0);
    CheckDivideScaler(v2, s1);
    CheckDivideScaler(v2, s2);
    CheckDivideScaler(v2, s3);
    CheckDivideScaler(v2, s0);
    CheckDivideScaler(v3, s1);
    CheckDivideScaler(v3, s2);
    CheckDivideScaler(v3, s3);
    CheckDivideScaler(v3, s0);
    CheckDivideScaler(v4, s1);
    CheckDivideScaler(v4, s2);
    CheckDivideScaler(v4, s3);
    CheckDivideScaler(v4, s0);
    CheckDivideScaler(v0, s3);
    CheckDivideScaler(v0, s0);
}

template <class T, class U>
void CheckDivideVector(const Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Divide(v2);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Divide(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division as the second operand is empty.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Dimension() % v2.Dimension() != 0)
    {
        EXPECT_THROW(
            try {
                v1.Divide(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division. Expected the dimension of the "
                      "second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    bool hasZero = false;
    for (std::size_t i = 0; i < v2.Size(); i++)
        if (v2[i] == 0)
        {
            hasZero = true;
            break;
        }
    if (!hasZero)
    {
        const auto result = v1.Divide(v2);
        for (std::size_t i = 0; i < result.Size(); i++)
            EXPECT_DOUBLE_EQ(v1[i] / v2[i % v2.Dimension()], result[i]);
    }
    else
    {
        EXPECT_THROW(
            try {
                v1.Divide(v2);
            } catch (const Exceptions::DividedByZero &e) {
                std::stringstream ss;
                ss << "Division by zero occurred.\n";
                ss << "Vector: Expect none of the element of the second operand to be 0 when performing"
                      "element-wise division.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::DividedByZero);
        return;
    }
}

TEST(Vector, DivideVector)
{
    Vector<int> v1({64, -13, 943});
    Vector<int> v2({269, -34, 43, 283, 364, -323, 734, 849});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.3f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.1, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<float> badV1({-2.124f, 23.2f, -4.32f, 0.f, 1.04f, 0.f, 32.3f, -49.f, 23.43f});
    Vector<double> badV2({3.14, -1.24, -0.5576, -2.3, 0.0, 23.0, -7.5, 64.56, 1.23, 2.0});
    Vector<int> v0;
    CheckDivideVector(v1, v2);
    CheckDivideVector(v2, v1);
    CheckDivideVector(v1, v3);
    CheckDivideVector(v3, v1);
    CheckDivideVector(v1, v4);
    CheckDivideVector(v4, v1);
    CheckDivideVector(v0, v1);
    CheckDivideVector(v1, v0);
    CheckDivideVector(v1, badV1);
    CheckDivideVector(v1, badV2);
    CheckDivideVector(v0, v0);
}

template <class T, class Scaler>
void CheckOperatorDivideScaler(const Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 / s;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (s == 0)
    {
        EXPECT_THROW(
            try {
                v1 / s;
            } catch (const Exceptions::DividedByZero &e) {
                std::stringstream ss;
                ss << "Division by zero occurred.\n";
                ss << "Vector: Cannot perform element-wise division as the second operand is 0.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::DividedByZero);
        return;
    }
    const auto result = v1 / s;
    for (std::size_t i = 0; i < v1.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i] / s, result[i]);
}

TEST(Vector, OperatorDivideScaler)
{
    Vector<int> v1({-4542, 34856, 7435, 438, -2594});
    Vector<int> v2({96, -234, 34534, 89063, 24189, -2856, 6, 805325, 934});
    Vector<float> v3({-5636.1454f, 243.2f, -582.32f, 874.3f, 23.234f, 1540.f, 332.3f, 6800450.f, 23.34532f});
    Vector<double> v4({23.435, -1.24454, -923.55676, -964.3, 0.0, 23.0324, -7.45455, 0.4485, 1.2323, 2.3423});
    Vector<int> v0;
    const int s1 = -12;
    const float s2 = 25873.631415f;
    const double s3 = 543.885644345;
    const float s0 = 0.f;
    CheckOperatorDivideScaler(v1, s1);
    CheckOperatorDivideScaler(v1, s2);
    CheckOperatorDivideScaler(v1, s3);
    CheckOperatorDivideScaler(v1, s0);
    CheckOperatorDivideScaler(v2, s1);
    CheckOperatorDivideScaler(v2, s2);
    CheckOperatorDivideScaler(v2, s3);
    CheckOperatorDivideScaler(v2, s0);
    CheckOperatorDivideScaler(v3, s1);
    CheckOperatorDivideScaler(v3, s2);
    CheckOperatorDivideScaler(v3, s3);
    CheckOperatorDivideScaler(v3, s0);
    CheckOperatorDivideScaler(v4, s1);
    CheckOperatorDivideScaler(v4, s2);
    CheckOperatorDivideScaler(v4, s3);
    CheckOperatorDivideScaler(v4, s0);
    CheckOperatorDivideScaler(v0, s3);
    CheckOperatorDivideScaler(v0, s0);
}

template <class T, class U>
void CheckOperatorDivideVector(const Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1 / v2;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1 / v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division as the second operand is empty.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Dimension() % v2.Dimension() != 0)
    {
        EXPECT_THROW(
            try {
                v1 / v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division. Expected the dimension of the "
                      "second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    bool hasZero = false;
    for (std::size_t i = 0; i < v2.Size(); i++)
        if (v2[i] == 0)
        {
            hasZero = true;
            break;
        }
    if (!hasZero)
    {
        const auto result = v1 / v2;
        for (std::size_t i = 0; i < result.Size(); i++)
            EXPECT_DOUBLE_EQ(v1[i] / v2[i % v2.Dimension()], result[i]);
    }
    else
    {
        EXPECT_THROW(
            try {
                v1 / v2;
            } catch (const Exceptions::DividedByZero &e) {
                std::stringstream ss;
                ss << "Division by zero occurred.\n";
                ss << "Vector: Expect none of the element of the second operand to be 0 when performing"
                      "element-wise division.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::DividedByZero);
        return;
    }
}

TEST(Vector, OperatorDivideVector)
{
    Vector<int> v1({64, -13, 943});
    Vector<int> v2({269, -34, 43, 283, 364, -323, 734, 849});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.3f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.1, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<float> badV1({-2.124f, 23.2f, -4.32f, 0.f, 1.04f, 0.f, 32.3f, -49.f, 23.43f});
    Vector<double> badV2({3.14, -1.24, -0.5576, -2.3, 0.0, 23.0, -7.5, 64.56, 1.23, 2.0});
    Vector<int> v0;
    CheckOperatorDivideVector(v1, v2);
    CheckOperatorDivideVector(v2, v1);
    CheckOperatorDivideVector(v1, v3);
    CheckOperatorDivideVector(v3, v1);
    CheckOperatorDivideVector(v1, v4);
    CheckOperatorDivideVector(v4, v1);
    CheckOperatorDivideVector(v0, v1);
    CheckOperatorDivideVector(v1, v0);
    CheckOperatorDivideVector(v1, badV1);
    CheckOperatorDivideVector(v1, badV2);
    CheckOperatorDivideVector(v0, v0);
}

template <class T, class Scaler>
void CheckOperatorDivideAssignmentScaler(Vector<T> &v1, const Scaler &s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 /= s;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (s == 0)
    {
        EXPECT_THROW(
            try {
                v1 /= s;
            } catch (const Exceptions::DividedByZero &e) {
                std::stringstream ss;
                ss << "Division by zero occurred.\n";
                ss << "Vector: Cannot perform element-wise division as the second operand is 0.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::DividedByZero);
        return;
    }
    const auto v1Copy = v1;
    v1 /= s;
    for (std::size_t i = 0; i < v1.Dimension(); i++)
        EXPECT_DOUBLE_EQ(v1[i], T(v1Copy[i] / s));
}

TEST(Vector, OperatorDivideAssignmentScaler)
{
    Vector<int> v1({-34, 243, -7435, 4554, -4});
    Vector<int> v2({96, -234, 12, -43, 56, -89, 6, 64, 934});
    Vector<float> v3({-103.1454f, 13.2f, -75.32f, 74.3f, -23.234f, 67.f, 53.3f, 434.f, 23.565});
    Vector<double> v4({23.435, -1.24454, -421.55676, -403.3, 324.0, 23.0324, -7.45455, 0.4485, 71.756, 42.3423});
    Vector<int> v0;
    const int s1 = -234;
    const float s2 = 34.4378;
    const double s3 = 905.2345;
    const double s0 = 0.0;
    CheckOperatorDivideAssignmentScaler(v1, s1);
    CheckOperatorDivideAssignmentScaler(v1, s2);
    CheckOperatorDivideAssignmentScaler(v1, s3);
    CheckOperatorDivideAssignmentScaler(v1, s0);
    CheckOperatorDivideAssignmentScaler(v2, s1);
    CheckOperatorDivideAssignmentScaler(v2, s2);
    CheckOperatorDivideAssignmentScaler(v2, s3);
    CheckOperatorDivideAssignmentScaler(v2, s0);
    CheckOperatorDivideAssignmentScaler(v3, s1);
    CheckOperatorDivideAssignmentScaler(v3, s2);
    CheckOperatorDivideAssignmentScaler(v3, s3);
    CheckOperatorDivideAssignmentScaler(v3, s0);
    CheckOperatorDivideAssignmentScaler(v4, s1);
    CheckOperatorDivideAssignmentScaler(v4, s2);
    CheckOperatorDivideAssignmentScaler(v4, s3);
    CheckOperatorDivideAssignmentScaler(v4, s0);
    CheckOperatorDivideAssignmentScaler(v0, s3);
    CheckOperatorDivideAssignmentScaler(v0, s0);
}

template <class T, class U>
void CheckOperatorDivideAssignmentVector(Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 /= v2;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Size() == 0)
    {
        EXPECT_THROW(
            try {
                v1 /= v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division when the second operand is empty.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Size() % v2.Size() != 0)
    {
        EXPECT_THROW(
            try {
                v1 /= v2;
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division. Expected the dimension of the "
                      "second operand to be a factor of that of the first operand.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    const auto v1Copy = v1;
    v1 /= v2;
    for (std::size_t i = 0; i < v1Copy.Size(); i++)
        EXPECT_DOUBLE_EQ(v1[i], T(v1Copy[i] / v2[i % v2.Dimension()]));
}

TEST(Vector, OperatorDivideAssignmentVector)
{
    Vector<int> v1({64, -13, 943});
    Vector<int> v2({269, -34, 43, 283, 364, -323, 734, 849});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.3f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.1, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<float> badV1({-2.124f, 23.2f, -4.32f, 0.f, 1.04f, 0.f, 32.3f, -49.f, 23.43f});
    Vector<double> badV2({3.14, -1.24, -0.5576, -2.3, 0.0, 23.0, -7.5, 64.56, 1.23, 2.0});
    Vector<int> v0;
    CheckOperatorDivideAssignmentVector(v1, v2);
    CheckOperatorDivideAssignmentVector(v2, v1);
    CheckOperatorDivideAssignmentVector(v1, v3);
    CheckOperatorDivideAssignmentVector(v3, v1);
    CheckOperatorDivideAssignmentVector(v1, v4);
    CheckOperatorDivideAssignmentVector(v4, v1);
    CheckOperatorDivideAssignmentVector(v0, v1);
    CheckOperatorDivideAssignmentVector(v1, v0);
    CheckOperatorDivideAssignmentVector(v1, badV1);
    CheckOperatorDivideAssignmentVector(v1, badV2);
    CheckOperatorDivideAssignmentVector(v0, v0);
}

template <class T, class U>
void CheckDotProduct(const Vector<T> &v1, const Vector<U> &v2)
{
    if (v1.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Dot(v2);
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform dot product on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    else if (v2.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v1.Dot(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform dot product when the second operand is empty.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    else if (v1.Dimension() != v2.Dimension())
    {
        EXPECT_THROW(
            try {
                v1.Dot(v2);
            } catch (const Exceptions::InvalidArgument &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform dot product on vectors with different dimensions.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::InvalidArgument);
        return;
    }
    double result = 0;
    for (std::size_t i = 0; i < v1.Dimension(); i++)
        result += v1[i] * v2[i];
    EXPECT_DOUBLE_EQ(result, v1.Dot(v2));
}

TEST(Vector, DotProduct)
{
    Vector<int> i1({4, 34, -23, 43, -69});
    Vector<int> i2({3, 54, 294, 984, 0});
    Vector<int> badI1;
    Vector<int> badI2({3, 54, 294, 984, 0, 213});
    Vector<float> f1({4, 34, -23, 43, -69});
    Vector<float> f2({3, 54, 294, 984, 0});
    Vector<float> d1({4, 34, -23, 43, -69});
    Vector<float> d2({3, 54, 294, 984, 0});
    CheckDotProduct(i1, i2);
    CheckDotProduct(i1, badI1);
    CheckDotProduct(i1, badI2);
    CheckDotProduct(i1, f1);
    CheckDotProduct(f1, f2);
    CheckDotProduct(f1, f1);
    CheckDotProduct(d1, d2);
}

template <class T>
void CheckNormalized(const Vector<T> &v)
{
    if (v.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v.template Normalized<double>();
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform normalization on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto len = v.template Length<double>();
    if (len == 0)
    {
        EXPECT_THROW(
            try {
                v.template Normalized<double>();
            } catch (const Exceptions::DividedByZero &e) {
                std::stringstream ss;
                ss << "Division by zero occurred.\n";
                ss << "Vector: Cannot normalize a zero vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::DividedByZero);
        return;
    }
    const auto normalized = v.template Normalized<double>();
    for (std::size_t i = 0; i < v.Dimension(); i++)
        EXPECT_DOUBLE_EQ(normalized[i], (double)v[i] / len);
}

TEST(Vector, Normalized)
{
    Vector<int> v1({64, -13, 943});
    Vector<int> v2({269, -34, 43, 283, 364, -323, 734, 849});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.3f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.1, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<float> badV1({0.f, 0.f, 0.f});
    Vector<double> badV2({0.0});
    Vector<int> v0;
    CheckNormalized(v1);
    CheckNormalized(v2);
    CheckNormalized(v3);
    CheckNormalized(v4);
    CheckNormalized(badV1);
    CheckNormalized(badV2);
    CheckNormalized(v0);
}

template <class T>
void CheckNormalize(Vector<T> &v)
{
    if (v.Dimension() == 0)
    {
        EXPECT_THROW(
            try {
                v.Normalize();
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform normalization on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto len = v.template Length<T>();
    if (len == 0)
    {
        EXPECT_THROW(
            try {
                v.Normalize();
            } catch (const Exceptions::DividedByZero &e) {
                std::stringstream ss;
                ss << "Division by zero occurred.\n";
                ss << "Vector: Cannot normalize a zero vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::DividedByZero);
        return;
    }
    const auto vCopy = v;
    v.Normalize();
    for (std::size_t i = 0; i < v.Dimension(); i++)
        EXPECT_DOUBLE_EQ(vCopy[i] / len, v[i]);
}

TEST(Vector, Normalize)
{
    Vector<float> v1({64.32f, -13.34f, 943.644f});
    Vector<double> v2({269.4, -34.64, 43.032, 283.34032, 364.0, -43.0, 4.023, 9.0});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.3f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.1, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<float> badV1({0.f, 0.f, 0.f});
    Vector<double> badV2({0.0});
    Vector<int> v0;
    CheckNormalize(v1);
    CheckNormalize(v2);
    CheckNormalize(v3);
    CheckNormalize(v4);
    CheckNormalize(badV1);
    CheckNormalize(badV2);
    CheckNormalize(v0);
}

template <class T>
void CheckSum(const Vector<T> &v)
{
    T sum = 0;
    for (std::size_t i = 0; i < v.Size(); i++)
        sum += v[i];
    EXPECT_NEAR(sum, v.Sum(), 0.0001);
}

TEST(Vector, Sum)
{
    Vector<float> v1({64.32f, -13.34f, 943.644f});
    Vector<double> v2({269.4, -34.64, 43.032, 283.34032, 364.0, -43.0, 4.023, 9.0});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.3f, 32.3f, -49.f, 23.43f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.1, 23.0, -7.5, 64.56, 1.23, 2.3423});
    Vector<int> v0;
    CheckSum(v1);
    CheckSum(v2);
    CheckSum(v3);
    CheckSum(v4);
    CheckSum(v0);
}

template <class T, class Func>
void CheckMap(const Vector<T> &v, Func &&f)
{
    const auto result = v.Map(f);
    for (std::size_t i = 0; i < v.Size(); i++)
        EXPECT_DOUBLE_EQ(f(v[i]), result[i]);
}

TEST(Vector, Map)
{
    Vector<int> v1({64, -133, 53});
    Vector<float> v2({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.3f, 32.3f, -49.f, 23.43f});
    Vector<double> v3({3.14, -2.0, 32.32, 8.235, 23.0, -7.5, 64.56, 1.23, 2.3423});
    const auto multTwo = [](const int e)
    { return e * 2; };
    const auto f1 = [](const float e)
    { return 2.f * e * e - 4.f * e + 435.23f; };
    const auto f2 = [](const double e)
    { return -3.2 * e * e * e + 2.3 * e * e + 9 * e + 23.4223; };
    CheckMap(v1, multTwo);
    CheckMap(v2, f1);
    CheckMap(v3, f2);
}

template <class T>
void CheckAsRawPointer(const Vector<T> &v)
{
    const auto p = v.AsRawPointer();
    for (std::size_t i = 0; i < v.Size(); i++)
        EXPECT_DOUBLE_EQ(v[i], p[i]);
}

TEST(Vector, AsRawPointer)
{
    Vector<int> v1({64, -133, 53});
    Vector<float> v2({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.3f, 32.3f, -49.f, 23.43f});
    Vector<double> v3({3.14, -2.0, 32.32, 8.235, 23.0, -7.5, 64.56, 1.23, 2.3423});
    CheckAsRawPointer(v1);
    CheckAsRawPointer(v2);
    CheckAsRawPointer(v3);
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

TEST(Vector, Combine)
{
    const std::vector<int> VECTOR_CONTENT_1({467, 235, 42, 692, 832, 11});
    const std::vector<int> VECTOR_CONTENT_2({1, 1, 2, 2, 4, 4});
    Vector<int> v1(VECTOR_CONTENT_1);
    Vector<int> v2(VECTOR_CONTENT_2);
    auto combined = Vector<int>::Combine({v1, v2});
    for (std::size_t i = 0; i < VECTOR_CONTENT_1.size(); i++)
        EXPECT_EQ(combined[i], VECTOR_CONTENT_1[i]);
    for (std::size_t i = 0; i < VECTOR_CONTENT_2.size(); i++)
        EXPECT_EQ(combined[VECTOR_CONTENT_1.size() + i], VECTOR_CONTENT_2[i]);
}

template <class T, class Scaler>
void CheckScalerVectorAddition(const Scaler &s, const Vector<T> &v)
{
    if (v.Size() == 0)
    {
        EXPECT_THROW(
            try {
                s + v;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform addition on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto result = s + v;
    for (std::size_t i = 0; i < v.Size(); i++)
        EXPECT_DOUBLE_EQ(s + v[i], result[i]);
}

TEST(Vector, ScalerVectorAddition)
{
    Vector<int> v1({43, -13});
    Vector<int> v2({96, -4, 99, 83, 48, -263, 34, 89});
    Vector<float> v3({-2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f});
    Vector<double> v4({3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423});
    Vector<int> v0;
    const int s1 = 32;
    const float s2 = 3.1415f;
    const double s3 = 56635.45245;
    CheckScalerVectorAddition(s1, v1);
    CheckScalerVectorAddition(s2, v1);
    CheckScalerVectorAddition(s3, v1);
    CheckScalerVectorAddition(s1, v2);
    CheckScalerVectorAddition(s2, v2);
    CheckScalerVectorAddition(s3, v2);
    CheckScalerVectorAddition(s1, v3);
    CheckScalerVectorAddition(s2, v3);
    CheckScalerVectorAddition(s3, v3);
    CheckScalerVectorAddition(s1, v4);
    CheckScalerVectorAddition(s2, v4);
    CheckScalerVectorAddition(s3, v4);
    CheckScalerVectorAddition(s3, v0);
}

template <class T, class Scaler>
void CheckScalerVectorSubtraction(const Scaler &s, const Vector<T> &v)
{
    if (v.Size() == 0)
    {
        EXPECT_THROW(
            try {
                s - v;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform subtraction on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto result = s - v;
    for (std::size_t i = 0; i < v.Size(); i++)
        EXPECT_DOUBLE_EQ(s - v[i], result[i]);
}

TEST(Vector, ScalerVectorSubtraction)
{
    Vector<int> v1({55, -19});
    Vector<int> v2({96, -4, 34, 83, 48, -286, 34, 325});
    Vector<float> v3({-2.1454f, 243.2f, -582.32f, 874.3f, 165.04f, 10.f, 332.3f, 0.f, 23.f});
    Vector<double> v4({23.435, -1.24454, -0.55676, -964.3, 0.0, 23.0, -7.45455, 0.4485, 1.2323, 2.3423});
    Vector<int> v0;
    const int s1 = 322342;
    const float s2 = 25873.1415f;
    const double s3 = 543.5644345;
    CheckScalerVectorSubtraction(s1, v1);
    CheckScalerVectorSubtraction(s2, v1);
    CheckScalerVectorSubtraction(s3, v1);
    CheckScalerVectorSubtraction(s1, v2);
    CheckScalerVectorSubtraction(s2, v2);
    CheckScalerVectorSubtraction(s3, v2);
    CheckScalerVectorSubtraction(s1, v3);
    CheckScalerVectorSubtraction(s2, v3);
    CheckScalerVectorSubtraction(s3, v3);
    CheckScalerVectorSubtraction(s1, v4);
    CheckScalerVectorSubtraction(s2, v4);
    CheckScalerVectorSubtraction(s3, v4);
    CheckScalerVectorSubtraction(s1, v0);
}

template <class T, class Scaler>
void CheckScalerVectorMultiplication(const Scaler &s, const Vector<T> &v)
{
    if (v.Size() == 0)
    {
        EXPECT_THROW(
            try {
                s *v;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform scaling on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    const auto result = s * v;
    for (std::size_t i = 0; i < v.Dimension(); i++)
        EXPECT_DOUBLE_EQ(s * v[i], result[i]);
}

TEST(Vector, ScalerVectorMultiplication)
{
    Vector<int> v1({-34, 243, -7435, 4554, -4});
    Vector<int> v2({96, -234, 12, -43, 56, -89, 6, 64, 934});
    Vector<float> v3({-103.1454f, 13.2f, -75.32f, 74.3f, -23.234f, 67.f, 53.3f, 434.f, 23.565});
    Vector<double> v4({23.435, -1.24454, -421.55676, -403.3, 324.0, 23.0324, -7.45455, 0.4485, 71.756, 42.3423});
    Vector<int> v0;
    const int s1 = -234;
    const float s2 = 34.4378;
    const double s3 = 905.2345;
    CheckScalerVectorMultiplication(s1, v1);
    CheckScalerVectorMultiplication(s2, v1);
    CheckScalerVectorMultiplication(s3, v1);
    CheckScalerVectorMultiplication(s1, v2);
    CheckScalerVectorMultiplication(s2, v2);
    CheckScalerVectorMultiplication(s3, v2);
    CheckScalerVectorMultiplication(s1, v3);
    CheckScalerVectorMultiplication(s2, v3);
    CheckScalerVectorMultiplication(s3, v3);
    CheckScalerVectorMultiplication(s1, v4);
    CheckScalerVectorMultiplication(s2, v4);
    CheckScalerVectorMultiplication(s3, v4);
    CheckScalerVectorMultiplication(s3, v0);
}

template <class T, class Scaler>
void CheckScalerVectorDivision(const Scaler &s, const Vector<T> &v)
{
    if (v.Size() == 0)
    {
        EXPECT_THROW(
            try {
                s / v;
            } catch (const Exceptions::EmptyVector &e) {
                std::stringstream ss;
                ss << "Vector: Cannot perform element-wise division on an empty vector.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::EmptyVector);
        return;
    }
    bool hasZero = false;
    for (std::size_t i = 0; i < v.Dimension(); i++)
        if (v[i] == 0)
        {
            hasZero = true;
            break;
        }
    if (hasZero)
    {
        EXPECT_THROW(
            try {
                s / v;
            } catch (const Exceptions::DividedByZero &e) {
                std::stringstream ss;
                ss << "Division by zero occurred.\nVector: Expect none of the elements of the second operand to be 0 when performing"
                      "element-wise division.";
                EXPECT_TRUE(e.what() == ss.str());
                throw e;
            },
            Exceptions::DividedByZero);
        return;
    }
    const auto result = s / v;
    for (std::size_t i = 0; i < v.Size(); i++)
        EXPECT_DOUBLE_EQ(s / v[i], result[i]);
}

TEST(Vector, ScalerVectorDivision)
{
    Vector<int> v1({-4542, 34856, 7435, 438, -2594});
    Vector<int> v2({96, -234, 34534, 89063, 24189, -2856, 6, 805325, 934});
    Vector<float> v3({-5636.1454f, 243.2f, -582.32f, 874.3f, 23.234f, 1540.f, 332.3f, 6800450.f, 23.34532f});
    Vector<double> v4({23.435, -1.24454, -923.55676, -964.3, 0.0, 23.0324, -7.45455, 0.4485, 1.2323, 2.3423});
    Vector<int> v0;
    const int s1 = -12;
    const float s2 = 525873.631415f;
    const double s3 = 454453.885644345;
    const int s0 = 0;
    CheckScalerVectorDivision(s1, v1);
    CheckScalerVectorDivision(s2, v1);
    CheckScalerVectorDivision(s3, v1);
    CheckScalerVectorDivision(s0, v1);
    CheckScalerVectorDivision(s1, v2);
    CheckScalerVectorDivision(s2, v2);
    CheckScalerVectorDivision(s3, v2);
    CheckScalerVectorDivision(s0, v2);
    CheckScalerVectorDivision(s1, v3);
    CheckScalerVectorDivision(s2, v3);
    CheckScalerVectorDivision(s3, v3);
    CheckScalerVectorDivision(s0, v3);
    CheckScalerVectorDivision(s1, v4);
    CheckScalerVectorDivision(s2, v4);
    CheckScalerVectorDivision(s3, v4);
    CheckScalerVectorDivision(s0, v4);
    CheckScalerVectorDivision(s3, v0);
    CheckScalerVectorDivision(s0, v0);
}