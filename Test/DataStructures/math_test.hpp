#include "Math.hpp"
#include <gtest/gtest.h>


#include <cmath>

TEST(Math, Exponent) {
  for (double i = -100; i <= 100.0; i += 0.01) {
    const auto stdExp = exp(i);
    const auto mathExp = Math::Exponent(i);
    EXPECT_TRUE(std::abs(stdExp - mathExp) / stdExp < 0.001);
  }
}

TEST(Math, NaturalLog) {
  for (double i = 0.01; i <= 100.0; i += 0.01)
    EXPECT_NEAR(log(i), Math::NaturalLog(i), 0.00001);
}
