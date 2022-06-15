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
    Vector<int> t1;
    Vector<float> t2({1.f, 2.f, 3.f, 4.f});
    Vector<double> t3({32.3, 42.42, 53.75});
    EXPECT_EQ(t1.Dimension(), 0);
    EXPECT_EQ(t2.Dimension(), 4);
    EXPECT_EQ(t3.Dimension(), 3);
}

TEST(Vector, Length)
{
    Vector<int> t1 = Vector<int>::ZeroVector(10);
    EXPECT_FLOAT_EQ(t1.Length<float>(), 0.f);
    Vector<float> t2({3.f, 4.f});
    EXPECT_FLOAT_EQ(t2.Length<float>(), 5.f);
    Vector<float> t3({-3.f, 4.f});
    EXPECT_FLOAT_EQ(t3.Length<float>(), 5.f);
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
    Vector<int> t1 = Vector<int>::ZeroVector(10);
    EXPECT_FLOAT_EQ(t1.EuclideanNorm<float>(), 0.f);
    Vector<float> t2({3.f, 4.f});
    EXPECT_FLOAT_EQ(t2.EuclideanNorm<float>(), 5.f);
    Vector<float> t3({-3.f, 4.f});
    EXPECT_FLOAT_EQ(t3.EuclideanNorm<float>(), 5.f);
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
    Vector<float> t1({3.f, 4.f});
    CheckVectorLpNorm(t1, 2);
    Vector<int> t2({4, 5, 64, 2, 67, 32, 54});
    CheckVectorLpNorm(t2, 5);
    Vector<double> t3({23.345, 4.43, 5.345, 64.23, 32.56, 6.4537, 32.536, 54.4535, 2684.567});
    CheckVectorLpNorm(t3, 8);
    Vector<int> t0;
    CheckVectorLpNorm(t0, 32);
}

template <class T, class U>
void CheckVectorAddition(const Vector<T>& v1, const Vector<U>& v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try
            {
                v1.Add(v2);
            }
            catch (const Exceptions::EmptyVector& e)
            {
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
            try
            {
                v1.Add(v2);
            }
            catch (const Exceptions::InvalidArgument& e)
            {
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
            try
            {
                v1.Add(v2);
            }
            catch (const Exceptions::InvalidArgument& e)
            {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second vector to be a factor of that of the first.";
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
    Vector<int> v1({ 43, -13 });
    Vector<int> v2({ 96, -4, 99, 83, 48, -263, 34, 89 });
    Vector<float> v3({ -2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f });
    Vector<double> v4({ 3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423 });
    CheckVectorAddition(v1, v2);
    CheckVectorAddition(v2, v1);
    CheckVectorAddition(v1, v3);
    CheckVectorAddition(v3, v1);
    CheckVectorAddition(v1, v4);
    CheckVectorAddition(v4, v1);
}

template <class T, class Scaler>
void CheckScalerAddition(const Vector<T>& v1, const Scaler& s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try
            {
                v1.Add(s);
            }
            catch (const Exceptions::EmptyVector& e)
            {
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
    Vector<int> v1({ 43, -13 });
    Vector<int> v2({ 96, -4, 99, 83, 48, -263, 34, 89 });
    Vector<float> v3({ -2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f });
    Vector<double> v4({ 3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423 });
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
}

template <class T, class U>
void CheckOperatorPlusVector(const Vector<T>& v1, const Vector<U>& v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try
            {
                v1 + v2;
            }
            catch (const Exceptions::EmptyVector& e)
            {
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
            try
            {
                v1 + v2;
            }
            catch (const Exceptions::InvalidArgument& e)
            {
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
            try
            {
                v1 + v2;
            }
            catch (const Exceptions::InvalidArgument& e)
            {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second vector to be a factor of that of the first.";
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
    Vector<int> v1({ 43, -13 });
    Vector<int> v2({ 96, -4, 99, 83, 48, -263, 34, 89 });
    Vector<float> v3({ -2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f });
    Vector<double> v4({ 3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423 });
    CheckOperatorPlusVector(v1, v2);
    CheckOperatorPlusVector(v2, v1);
    CheckOperatorPlusVector(v1, v3);
    CheckOperatorPlusVector(v3, v1);
    CheckOperatorPlusVector(v1, v4);
    CheckOperatorPlusVector(v4, v1);
}

template <class T, class Scaler>
void CheckOperatorPlusScaler(const Vector<T>& v1, const Scaler& s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try
            {
                v1 + s;
            }
            catch (const Exceptions::EmptyVector& e)
            {
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
    Vector<int> v1({ 43, -13 });
    Vector<int> v2({ 96, -4, 99, 83, 48, -263, 34, 89 });
    Vector<float> v3({ -2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f });
    Vector<double> v4({ 3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423 });
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
}

template <class T, class U>
void CheckOperatorPlusAssignmentVector(Vector<T>& v1, const Vector<U>& v2)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try
            {
                v1 += v2;
            }
            catch (const Exceptions::EmptyVector& e)
            {
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
            try
            {
                v1 += v2;
            }
            catch (const Exceptions::InvalidArgument& e)
            {
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
            try
            {
                v1 += v2;
            }
            catch (const Exceptions::InvalidArgument& e)
            {
                std::stringstream ss;
                ss << "Vector: Expected the dimension of the second vector to be a factor of that of the first.";
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
    Vector<int> v1({ 43, -13 });
    Vector<int> v2({ 96, -4, 99, 83, 48, -263, 34, 89 });
    Vector<float> v3({ -2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f });
    Vector<double> v4({ 3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423 });
    CheckOperatorPlusAssignmentVector(v1, v2);
    CheckOperatorPlusAssignmentVector(v2, v1);
    CheckOperatorPlusAssignmentVector(v1, v3);
    CheckOperatorPlusAssignmentVector(v3, v1);
    CheckOperatorPlusAssignmentVector(v1, v4);
    CheckOperatorPlusAssignmentVector(v4, v1);
}

template <class T, class Scaler>
void CheckOperatorPlusAssignmentScaler(Vector<T>& v1, const Scaler& s)
{
    if (v1.Size() == 0)
    {
        EXPECT_THROW(
            try
            {
                v1 += s;
            }
            catch (const Exceptions::EmptyVector& e)
            {
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
    Vector<int> v1({ 43, -13 });
    Vector<int> v2({ 96, -4, 99, 83, 48, -263, 34, 89 });
    Vector<float> v3({ -2.124f, 23.2f, -82.32f, 84.3f, 1.04f, 0.f, 32.3f, -9.f, 23.f });
    Vector<double> v4({ 3.14, -1.24, -0.5576, -94.3, 0.0, 23.0, -7.5, 0.85, 1.23, 2.3423 });
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