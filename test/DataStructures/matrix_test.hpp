#include <gtest/gtest.h>

#include "Matrix.hpp"

#define ZERO 0

using namespace DataStructures;

TEST(Matrix, EmptyConstructor) {
  Matrix<int> empty;
  EXPECT_EQ(empty.Size(), ZERO);
  EXPECT_TRUE(empty.IsEmpty());
}

enum class ArithmeticType {
    Addition,
    Subtraction,
    Scaling,
    Division
};

template <class T, class U>
static void CheckMatrixArithmetic(const Matrix<T> &m1, const Matrix<U> &m2,
                                  ArithmeticType type) {
  Matrix<decltype(m1[0][0] + m2[0][0])> result;
  switch (type) {
  case ArithmeticType::Addition:
    result = m1 + m2;
    break;
  case ArithmeticType::Subtraction:
    result = m1 - m2;
    break;
  case ArithmeticType::Scaling:
    result = m1.Scale(m2);
    break;
  case ArithmeticType::Division:
    result = m1 / m2;
    break;
  default:
    break;
  }

  const auto m2Shape = m2.Shape();
  Matrix<decltype(m1[0][0] + m2[0][0])> expected(m1);
  for (size_t i = ZERO; i < expected.Shape()[0]; i++) {
    for (size_t j = ZERO; j < expected.Shape()[1]; j++) {
      switch (type) {
      case ArithmeticType::Addition:
        expected[i][j] += m2[i % m2Shape[0]][j % m2Shape[1]];
        break;
      case ArithmeticType::Subtraction:
        expected[i][j] -= m2[i % m2Shape[0]][j % m2Shape[1]];
        break;
      case ArithmeticType::Scaling:
        expected[i][j] *= m2[i % m2Shape[0]][j % m2Shape[1]];
        break;
      case ArithmeticType::Division:
        expected[i][j] /= m2[i % m2Shape[0]][j % m2Shape[1]];
        break;
      default:
        break;
      }
    }
  }

  for (size_t i = 0; i < expected.Shape()[0]; i++) {
    for (size_t j = 0; j < expected.Shape()[1]; j++) {
      EXPECT_DOUBLE_EQ(result[i][j], expected[i][j]);
    }
  }

  result = m1;
  switch (type) {
  case ArithmeticType::Addition:
    result += m2;
    break;
  case ArithmeticType::Subtraction:
    result -= m2;
    break;
  case ArithmeticType::Scaling:
    result *= m2;
    break;
  case ArithmeticType::Division:
    result /= m2;
    break;
  default:
    break;
  }

  for (size_t i = ZERO; i < expected.Shape()[0]; i++) {
    for (size_t j = ZERO; j < expected.Shape()[1]; j++) {
      EXPECT_DOUBLE_EQ(result[i][j], expected[i][j]);
    }
  }
}

TEST(Matrix, Addition) {
  Matrix<int> m1({{1, 2, 3}, {4, 5, 6}});
  Matrix<int> m2({98, 65});
  CheckMatrixArithmetic(m1, m2, ArithmeticType::Addition);
}

TEST(Matrix, Subtraction) {
  Matrix<int> m1({{2, 4, 89}, {45, 0, 65}});
  Matrix<int> m2({98, 65});
  CheckMatrixArithmetic(m1, m2, ArithmeticType::Subtraction);
}

TEST(Matrix, Scaling) {
  Matrix<float> m1({{1.5f, 2.f, 3.f}, {4.f, 5.2f, 6.f}});
  Matrix<int> m2({98, 65});
  CheckMatrixArithmetic(m1, m2, ArithmeticType::Scaling);
}

TEST(Matrix, Division) {
  Matrix<int> m1({{1, 2, 3}, {4, 5, 6}});
  Matrix<double> m2({98.23, 65.87});
  CheckMatrixArithmetic(m1, m2, ArithmeticType::Division);
}

template <class T, class U>
static void CheckMatrixMultiplication(const Matrix<T> &m1,
                                      const Matrix<U> &m2) {
  const auto result1 = m1 * m2;
  const auto result2 = m1.Multiply(m2);
  for (size_t i = ZERO; i < m1.Shape()[0]; i++) {
    for (size_t j = ZERO; j < m2.Shape()[1]; j++) {
      EXPECT_EQ(result1[i][j], result2[i][j]);
    }
  }
  Matrix<decltype(m1[0][0] * m2[0][0])> expected(m1.Shape()[0], m2.Shape()[1]);
  for (size_t i = ZERO; i < expected.Shape()[0]; i++) {
    for (size_t j = ZERO; j < expected.Shape()[1]; j++) {
      for (size_t k = ZERO; k < m1.Shape()[1]; k++) {
        expected[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }

  for (size_t i = ZERO; i < m1.Shape()[0]; i++) {
    for (size_t j = ZERO; j < m2.Shape()[1]; j++) {
      EXPECT_EQ(result1[i][j], expected[i][j]);
    }
  }
}

TEST(Matrix, Multiplication) {
  Matrix<int> m1({{1, 2}, {4, 5}});
  Matrix<int> m2({{1, 2, 3}, {4, 5, 6}});
  CheckMatrixMultiplication(m1, m2);
}

template <class T> void CheckTranspose(const Matrix<T> &m) {
  auto mCopy = m;
  mCopy.Transpose();
  for (size_t i = ZERO; i < m.Shape()[0]; i++) {
    for (size_t j = ZERO; j < m.Shape()[1]; j++) {
      EXPECT_DOUBLE_EQ(m[i][j], mCopy[j][i]);
    }
  }
  mCopy = m.Transposed();
  for (size_t i = ZERO; i < m.Shape()[0]; i++) {
    for (size_t j = ZERO; j < m.Shape()[1]; j++) {
      EXPECT_DOUBLE_EQ(m[i][j], mCopy[j][i]);
    }
  }
}

TEST(Matrix, Transpose) {
  Matrix<int> m1({{1, 2}, {4, 5}});
  Matrix<int> m2({{1, 2, 3}, {4, 5, 6}});
  CheckTranspose(m1);
  CheckTranspose(m2);
}

template <class T> void CheckSum(const Matrix<T> &m) {
  auto result = m.Sum();
  Matrix<T> expected(1, m.Shape()[1]);
  for (size_t i = ZERO; i < m.Shape()[0]; i++) {
    for (size_t j = ZERO; j < m.Shape()[1]; j++) {
      expected[0][j] += m[i][j];
    }
  }
  for (size_t j = ZERO; j < m.Shape()[1]; j++) {
    EXPECT_DOUBLE_EQ(result[0][j], expected[0][j]);
  }

  result = m.Sum(false);
  expected = Matrix<T>(m.Shape()[0], 1);
  for (size_t i = ZERO; i < m.Shape()[0]; i++) {
    for (size_t j = ZERO; j < m.Shape()[1]; j++) {
      expected[i][0] += m[i][j];
    }
  }
  for (size_t i = ZERO; i < m.Shape()[0]; i++) {
    EXPECT_DOUBLE_EQ(result[i][0], expected[i][0]);
  }
}

TEST(Matrix, Sum) {
  Matrix<int> m1({{1, 2}, {4, 5}});
  Matrix<int> m2({{1, 2, 3}, {4, 5, 6}});
  CheckSum(m1);
  CheckSum(m2);
}

template <class T> void CheckSumAll(const Matrix<T>& m) {
  const auto result = m.SumAll();
  T expected = 0;
  for (size_t i = ZERO; i < m.Shape()[0]; i++) {
    for (size_t j = ZERO; j < m.Shape()[1]; j++) {
      expected += m[i][j];
    }
  }
  EXPECT_DOUBLE_EQ(result, expected);
}

TEST(Matrix, SumAll) {
  Matrix<int> m1({{1, 2}, {4, 5}});
  Matrix<double> m2({{153.45, 2.19, 38.65}, {4.23, 547.25, 6.97}});
  CheckSumAll(m1);
  CheckSumAll(m2);
}
