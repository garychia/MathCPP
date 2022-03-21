#include <gtest/gtest.h>

#include "../vector.hpp"

using namespace Math;

#define TEST_VECTOR_INT(v, a, b, c) \
    EXPECT_EQ((v).x, a);        \
    EXPECT_EQ((v).y, b);        \
    EXPECT_EQ((v).z, c);

#define TEST_VECTOR_FLOAT(v, a, b, c) \
    EXPECT_FLOAT_EQ((v).x, a);        \
    EXPECT_FLOAT_EQ((v).y, b);        \
    EXPECT_FLOAT_EQ((v).z, c);

#define TEST_VECTOR_DOUBLE(v, a, b, c) \
    EXPECT_DOUBLE_EQ((v).x, a);        \
    EXPECT_DOUBLE_EQ((v).y, b);        \
    EXPECT_DOUBLE_EQ((v).z, c);

TEST(Vector3D, Vector3DConstructor)
{
    Vector3D<int> v1Int(1, -10, 100);
    Vector3D<int> v2Int(1, -20);
    Vector3D<int> v3Int(100);

    Vector3D<float> v1Float(1.f, -10.f, 100.f);
    Vector3D<float> v2Float(1.f, -10.f);
    Vector3D<float> v3Float(1.f);

    Vector3D<double> v1Double(3.14, -6.89, 34.34);
    Vector3D<double> v2Double(3.14, -6.89);
    Vector3D<double> v3Double(3.14);

    TEST_VECTOR_INT(v1Int, 1, -10, 100)
    TEST_VECTOR_INT(v2Int, 1, -20, 0)
    TEST_VECTOR_INT(v3Int, 100, 0, 0)

    TEST_VECTOR_FLOAT(v1Float, 1.f, -10.f, 100.f)
    TEST_VECTOR_FLOAT(v2Float, 1.f, -10.f, 0.f)
    TEST_VECTOR_FLOAT(v3Float, 1.f, 0.f, 0.f)

    TEST_VECTOR_DOUBLE(v1Double, 3.14, -6.89, 34.34)
    TEST_VECTOR_DOUBLE(v2Double, 3.14, -6.89, 0.0)
    TEST_VECTOR_DOUBLE(v3Double, 3.14, 0.0, 0.0)
}

TEST(Vector3D, VectorAddition)
{
    Vector3D<int> v1Int(1, -10, 100);
    Vector3D<int> v2Int(1, -12, 932);
    Vector3D<int> v3Int = v1Int + v2Int;
    TEST_VECTOR_INT(v3Int,
                    v1Int.x + v2Int.x,
                    v1Int.y + v2Int.y,
                    v1Int.z + v2Int.z)
    v3Int = v1Int.Add(v2Int);
    TEST_VECTOR_INT(v3Int,
                    v1Int.x + v2Int.x,
                    v1Int.y + v2Int.y,
                    v1Int.z + v2Int.z)
    v3Int += v1Int;
    TEST_VECTOR_INT(v3Int,
                    v1Int.x * 2 + v2Int.x,
                    v1Int.y * 2 + v2Int.y,
                    v1Int.z * 2 + v2Int.z)

    Vector3D<float> v1Float(3.4f, -1320.23f, 23.f);
    Vector3D<float> v2Float(231.f, -23.f, 932.f);
    TEST_VECTOR_FLOAT(
        v1Float + v2Float,
        v1Float.x + v2Float.x,
        v1Float.y + v2Float.y,
        v1Float.z + v2Float.z)
    TEST_VECTOR_FLOAT(
        v1Float.Add(v2Float),
        v1Float.x + v2Float.x,
        v1Float.y + v2Float.y,
        v1Float.z + v2Float.z)

    Vector3D<double> v1Double(31.4f, -20.23f, 23.f);
    Vector3D<double> v2Double(31.f, -23.f, 93.f);
    TEST_VECTOR_FLOAT(
        v1Double + v2Double,
        v1Double.x + v2Double.x,
        v1Double.y + v2Double.y,
        v1Double.z + v2Double.z)
    TEST_VECTOR_FLOAT(
        v1Double.Add(v2Double),
        v1Double.x + v2Double.x,
        v1Double.y + v2Double.y,
        v1Double.z + v2Double.z)
}