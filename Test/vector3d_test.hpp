#include <gtest/gtest.h>

#include "../DataStructures/Vector3D.hpp"

using namespace DataStructure;

#define TEST_VECTOR_INT(v, a, b, c) \
    EXPECT_EQ((v).X(), a);            \
    EXPECT_EQ((v).Y(), b);            \
    EXPECT_EQ((v).Z(), c);

#define TEST_VECTOR_FLOAT(v, a, b, c) \
    EXPECT_FLOAT_EQ((v).X(), a);        \
    EXPECT_FLOAT_EQ((v).Y(), b);        \
    EXPECT_FLOAT_EQ((v).Z(), c);

#define TEST_VECTOR_DOUBLE(v, a, b, c) \
    EXPECT_DOUBLE_EQ((v).X(), a);        \
    EXPECT_DOUBLE_EQ((v).Y(), b);        \
    EXPECT_DOUBLE_EQ((v).Z(), c);

#define TEST_VECTOR_INT_EQUAL(v1, v2) \
    TEST_VECTOR_INT(v1, (v2).X(), (v2).Y(), (v2).Z())

#define TEST_VECTOR_FLOAT_EQUAL(v1, v2) \
    TEST_VECTOR_FLOAT(v1, (v2).X(), (v2).Y(), (v2).Z())

#define TEST_VECTOR_DOUBLE_EQUAL(v1, v2) \
    TEST_VECTOR_DOUBLE(v1, (v2).X(), (v2).Y(), (v2).Z())

#define TEST_VECTOR_INT_OPERATOR(v1, v2, op) \
    TEST_VECTOR_INT((v1)op(v2), (v1).X() op(v2).X(), (v1).Y() op(v2).Y(), (v1).Z() op(v2).Z())

#define TEST_VECTOR_FLOAT_OPERATION(v1, v2, op) \
    TEST_VECTOR_FLOAT((v1)op(v2), (v1).X() op(v2).X(), (v1).Y() op(v2).Y(), (v1).Z() op(v2).Z())

#define TEST_VECTOR_DOUBLE_OPERATION(v1, v2, op) \
    TEST_VECTOR_DOUBLE((v1)op(v2), (v1).X() op(v2).X(), (v1).Y() op(v2).Y(), (v1).Z() op(v2).Z())

#define TEST_VECTOR_INT_ARITHMETIC_FUNCTION(v1, v2, f, op) \
    TEST_VECTOR_INT((v1).f(v2), (v1).X() op(v2).X(), (v1).Y() op(v2).Y(), (v1).Z() op(v2).Z())

#define TEST_VECTOR_FLOAT_ARITHMETIC_FUNCTION(v1, v2, f, op) \
    TEST_VECTOR_FLOAT((v1).f(v2), (v1).X() op(v2).X(), (v1).Y() op(v2).Y(), (v1).Z() op(v2).Z())

#define TEST_VECTOR_DOUBLE_ARITHMETIC_FUNCTION(v1, v2, f, op) \
    TEST_VECTOR_DOUBLE((v1).f(v2), (v1).X() op(v2).X(), (v1).Y() op(v2).Y(), (v1).Z() op(v2).Z())

#define TEST_VECTOR_INT_ADDITION(v1, v2) \
    TEST_VECTOR_INT_OPERATOR(v1, v2, +)  \
    TEST_VECTOR_INT_ARITHMETIC_FUNCTION(v1, v2, Add, +)

#define TEST_VECTOR_FLOAT_ADDITION(v1, v2) \
    TEST_VECTOR_FLOAT_OPERATION(v1, v2, +) \
    TEST_VECTOR_FLOAT_ARITHMETIC_FUNCTION(v1, v2, Add, +)

#define TEST_VECTOR_DOUBLE_ADDITION(v1, v2) \
    TEST_VECTOR_DOUBLE_OPERATION(v1, v2, +) \
    TEST_VECTOR_DOUBLE_ARITHMETIC_FUNCTION(v1, v2, Add, +)

#define TEST_VECTOR_INT_SUBTRACTION(v1, v2) \
    TEST_VECTOR_INT_OPERATOR(v1, v2, -)     \
    TEST_VECTOR_INT_ARITHMETIC_FUNCTION(v1, v2, Minus, -)

#define TEST_VECTOR_FLOAT_SUBTRACTION(v1, v2) \
    TEST_VECTOR_FLOAT_OPERATION(v1, v2, -)    \
    TEST_VECTOR_FLOAT_ARITHMETIC_FUNCTION(v1, v2, Minus, -)

#define TEST_VECTOR_DOUBLE_SUBTRACTION(v1, v2) \
    TEST_VECTOR_DOUBLE_OPERATION(v1, v2, -)    \
    TEST_VECTOR_DOUBLE_ARITHMETIC_FUNCTION(v1, v2, Minus, -)

#define TEST_VECTOR_INT_SCALING(v, s)                    \
    TEST_VECTOR_INT((v)*s, (v).X() *s, (v).Y() *s, (v).Z() *s) \
    TEST_VECTOR_INT((v).Scale(s), (v).X() *s, (v).Y() *s, (v).Z() *s)

#define TEST_VECTOR_FLOAT_SCALING(v, s)                    \
    TEST_VECTOR_FLOAT((v)*s, (v).X() *s, (v).Y() *s, (v).Z() *s) \
    TEST_VECTOR_FLOAT((v).Scale(s), (v).X() *s, (v).Y() *s, (v).Z() *s)

#define TEST_VECTOR_DOUBLE_SCALING(v, s)                    \
    TEST_VECTOR_DOUBLE((v)*s, (v).X() *s, (v).Y() *s, (v).Z() *s) \
    TEST_VECTOR_DOUBLE((v).Scale(s), (v).X() *s, (v).Y() *s, (v).Z() *s)

#define TEST_VECTOR_INT_DIVISION(v, s)                        \
    TEST_VECTOR_INT((v) / s, (v).X() / s, (v).Y() / s, (v).Z() / s) \
    TEST_VECTOR_INT((v).Divide(s), (v).X() / s, (v).Y() / s, (v).Z() / s)

#define TEST_VECTOR_FLOAT_DIVISION(v, s)                        \
    TEST_VECTOR_FLOAT((v) / s, (v).X() / s, (v).Y() / s, (v).Z() / s) \
    TEST_VECTOR_FLOAT((v).Divide(s), (v).X() / s, (v).Y() / s, (v).Z() / s)

#define TEST_VECTOR_DOUBLE_DIVISION(v, s)                        \
    TEST_VECTOR_DOUBLE((v) / s, (v).X() / s, (v).Y() / s, (v).Z() / s) \
    TEST_VECTOR_DOUBLE((v).Divide(s), (v).X() / s, (v).Y() / s, (v).Z() / s)

#define TEST_VECTOR_INT_DOT_PRODUCT(v1, v2) \
    EXPECT_EQ((v1).Dot(v2), (v1).X() *(v2).X() + (v1).Y() * (v2).Y() + (v1).Z() * (v2).Z());

#define TEST_VECTOR_FLOAT_DOT_PRODUCT(v1, v2) \
    EXPECT_FLOAT_EQ((v1).Dot(v2), (v1).X() *(v2).X() + (v1).Y() * (v2).Y() + (v1).Z() * (v2).Z());

#define TEST_VECTOR_DOUBLE_DOT_PRODUCT(v1, v2) \
    EXPECT_DOUBLE_EQ((v1).Dot(v2), (v1).X() *(v2).X() + (v1).Y() * (v2).Y() + (v1).Z() * (v2).Z());

#define TEST_VECTOR_INT_CROSS_PRODUCT(v1, v2) \
    TEST_VECTOR_INT((v1).Cross(v2), (v1).Y() *(v2).Z() - (v1).Z() * (v2).Y(), (v1).Z() * (v2).X() - (v1).X() * (v2).Z(), (v1).X() * (v2).Y() - (v1).Y() * (v2).X())

#define TEST_VECTOR_FLOAT_CROSS_PRODUCT(v1, v2) \
    TEST_VECTOR_FLOAT((v1).Cross(v2), (v1).Y() *(v2).Z() - (v1).Z() * (v2).Y(), (v1).Z() * (v2).X() - (v1).X() * (v2).Z(), (v1).X() * (v2).Y() - (v1).Y() * (v2).X())

#define TEST_VECTOR_DOUBLE_CROSS_PRODUCT(v1, v2) \
    TEST_VECTOR_DOUBLE((v1).Cross(v2), (v1).Y() *(v2).Z() - (v1).Z() * (v2).Y(), (v1).Z() * (v2).X() - (v1).X() * (v2).Z(), (v1).X() * (v2).Y() - (v1).Y() * (v2).X())

TEST(Vector3D, Vector3DConstructor)
{
    Vector3D<int> v1Int(1, -10, 100);
    Vector3D<int> v2Int(1, -20);
    Vector3D<int> v3Int(100);
    Vector3D<int> zeroVInt;

    Vector3D<float> v1Float(1.f, -10.f, 100.f);
    Vector3D<float> v2Float(1.f, -10.f);
    Vector3D<float> v3Float(1.f);
    Vector3D<float> zeroVDouble;

    Vector3D<double> v1Double(3.14, -6.89, 34.34);
    Vector3D<double> v2Double(3.14, -6.89);
    Vector3D<double> v3Double(3.14);
    Vector3D<double> zeroVFloat;

    TEST_VECTOR_INT(v1Int, 1, -10, 100)
    TEST_VECTOR_INT(v2Int, 1, -20, 0)
    TEST_VECTOR_INT(v3Int, 100, 0, 0)
    TEST_VECTOR_INT(zeroVInt, 0, 0, 0)

    TEST_VECTOR_FLOAT(v1Float, 1.f, -10.f, 100.f)
    TEST_VECTOR_FLOAT(v2Float, 1.f, -10.f, 0.f)
    TEST_VECTOR_FLOAT(v3Float, 1.f, 0.f, 0.f)
    TEST_VECTOR_FLOAT(zeroVFloat, 0.f, 0.f, 0.f)

    TEST_VECTOR_DOUBLE(v1Double, 3.14, -6.89, 34.34)
    TEST_VECTOR_DOUBLE(v2Double, 3.14, -6.89, 0.0)
    TEST_VECTOR_DOUBLE(v3Double, 3.14, 0.0, 0.0)
    TEST_VECTOR_DOUBLE(zeroVDouble, 0.0, 0.0, 0.0)
}

TEST(Vector3D, VectorAddition)
{
    Vector3D<int> v1Int(1, -10, 100);
    Vector3D<int> v2Int(1, -12, 932);
    Vector3D<int> v3Int = v1Int + v2Int;
    TEST_VECTOR_INT_ADDITION(v1Int, v2Int)
    v1Int += v2Int;
    TEST_VECTOR_INT_EQUAL(v1Int, v3Int)

    Vector3D<float> v1Float(3.4f, -1320.23f, 23.f);
    Vector3D<float> v2Float(231.f, -23.f, 932.f);
    Vector3D<float> v3Float = v1Float + v2Float;
    TEST_VECTOR_FLOAT_ADDITION(v1Float, v2Float)
    v1Float += v2Float;
    TEST_VECTOR_FLOAT_EQUAL(v1Float, v3Float)

    Vector3D<double> v1Double(31.423, -20.233, 23.234);
    Vector3D<double> v2Double(31.54, -23.234, 93.76);
    Vector3D<double> v3Double = v1Double + v2Double;
    TEST_VECTOR_DOUBLE_ADDITION(v1Double, v2Double)
    v1Double += v2Double;
    TEST_VECTOR_DOUBLE_EQUAL(v1Double, v3Double)

    Vector3D<float> vIntFloatSum = v1Int + v1Float;
    TEST_VECTOR_FLOAT_EQUAL(v1Int + v1Float, vIntFloatSum)
    Vector3D<double> vDoubleFloatSum = v1Float + v1Double;
    TEST_VECTOR_DOUBLE_EQUAL(v1Float + v1Double, vDoubleFloatSum)
    Vector3D<double> vIntDoubleSum = v1Int + v1Double;
    TEST_VECTOR_DOUBLE_EQUAL(v1Int + v1Double, vIntDoubleSum)
}

TEST(Vector3D, VectorSubtraction)
{
    Vector3D<int> v1Int(1, -10, 100);
    Vector3D<int> v2Int(1, -12, 932);
    Vector3D<int> v3Int = v1Int - v2Int;
    TEST_VECTOR_INT_SUBTRACTION(v1Int, v2Int)
    v1Int -= v2Int;
    TEST_VECTOR_INT_EQUAL(v1Int, v3Int)

    Vector3D<float> v1Float(3.4f, -1320.23f, 23.f);
    Vector3D<float> v2Float(231.f, -23.f, 932.f);
    Vector3D<float> v3Float = v1Float - v2Float;
    TEST_VECTOR_FLOAT_SUBTRACTION(v1Float, v2Float)
    v1Float -= v2Float;
    TEST_VECTOR_FLOAT_EQUAL(v1Float, v3Float);

    Vector3D<double> v1Double(31.4, -20.23, 23.54);
    Vector3D<double> v2Double(31.123, -23.55, 93.94);
    Vector3D<double> v3Double = v1Double - v2Double;
    TEST_VECTOR_DOUBLE_SUBTRACTION(v1Double, v2Double)
    v1Double -= v2Double;
    TEST_VECTOR_DOUBLE_EQUAL(v1Double, v3Double);

    Vector3D<float> vIntFloatSubtract = v1Int - v1Float;
    TEST_VECTOR_FLOAT_EQUAL(v1Int - v1Float, vIntFloatSubtract)
    Vector3D<double> vDoubleFloatSubtract = v1Float - v1Double;
    TEST_VECTOR_DOUBLE_EQUAL(v1Float - v1Double, vDoubleFloatSubtract)
    Vector3D<double> vIntDoubleSubtract = v1Int - v1Double;
    TEST_VECTOR_DOUBLE_EQUAL(v1Int - v1Double, vIntDoubleSubtract)
}

TEST(Vector3D, VectorDivision)
{
    Vector3D<int> v1Int(1, -10, 100);
    Vector3D<int> v2Int = v1Int;
    TEST_VECTOR_INT_DIVISION(v1Int, 34)
    v1Int /= 2;
    TEST_VECTOR_INT_EQUAL(v1Int, v2Int / 2)

    Vector3D<float> v1Float(3.4f, -1320.23f, 23.f);
    Vector3D<float> v2Float = v1Float;
    TEST_VECTOR_FLOAT_DIVISION(v1Float, 324.4f)
    v1Float /= 10;
    TEST_VECTOR_FLOAT_EQUAL(v1Float, v2Float / 10)

    Vector3D<double> v1Double(31.4, -20.23, 23.234);
    Vector3D<double> v2Double = v1Double;
    TEST_VECTOR_DOUBLE_DIVISION(v1Double, 332.343)
    v1Double /= 3.14;
    TEST_VECTOR_DOUBLE_EQUAL(v1Double, v2Double / 3.14)
}

TEST(Vector3D, VectorDotProduct)
{
    Vector3D<int> v1Int(1, -10, 100);
    Vector3D<int> v2Int(32, 17, 34);
    TEST_VECTOR_INT_DOT_PRODUCT(v1Int, v2Int)

    Vector3D<float> v1Float(3.4f, -1230.23f, 23.f);
    Vector3D<float> v2Float(24.23f, -23.65f, -67.345f);
    TEST_VECTOR_FLOAT_DOT_PRODUCT(v1Float, v2Float)

    Vector3D<double> v1Double(31.4, -20.23, 23.234);
    Vector3D<double> v2Double(9.324, 94.2354, -234.32);
    TEST_VECTOR_DOUBLE_DOT_PRODUCT(v1Double, v2Double)
}

TEST(Vector3D, VectorCrossProduct)
{
    Vector3D<int> v1Int(1, -310, 45);
    Vector3D<int> v2Int(23, 56, 94);
    TEST_VECTOR_INT_CROSS_PRODUCT(v1Int, v2Int)

    Vector3D<float> v1Float(3.4f, -1230.23f, 23.f);
    Vector3D<float> v2Float(24.23f, -23.65f, -67.345f);
    TEST_VECTOR_FLOAT_CROSS_PRODUCT(v1Float, v2Float)

    Vector3D<double> v1Double(31.4, -20.23, 23.234);
    Vector3D<double> v2Double(9.324, 94.2354, -234.32);
    TEST_VECTOR_DOUBLE_CROSS_PRODUCT(v1Double, v2Double)
}

TEST(Vector3D, VectorToString)
{
    Vector3D<int> v1Int(1, -310, 45);
    Vector3D<int> v2Int(23, 56, 94);
    EXPECT_TRUE(v1Int.ToString() == "(1, -310, 45)" && v2Int.ToString() == "(23, 56, 94)");

    Vector3D<float> v1Float(3.4f, -1230.23f, 23.f);
    Vector3D<float> v2Float(24.23f, -23.65f, -67.345f);
    EXPECT_TRUE(v1Float.ToString() == "(3.4, -1230.23, 23)" && v2Float.ToString() == "(24.23, -23.65, -67.345)");

    Vector3D<double> v1Double(31.4, -20.23, 23.234);
    Vector3D<double> v2Double(9.324, 94.2354, -234.32);
    EXPECT_TRUE(v1Double.ToString() == "(31.4, -20.23, 23.234)" && v2Double.ToString() == "(9.324, 94.2354, -234.32)");
}