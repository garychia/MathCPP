#include <gtest/gtest.h>
#include "Math.hpp"

#include <cmath>

TEST(Math, Exponent)
{
    for (double i = -100; i <= 100.0; i += 0.01)
    {
        const auto stdExp = exp(i);
        const auto mathExp = Math::Exponent(i);
        const auto diff = stdExp - mathExp;
        EXPECT_TRUE(ABS(diff) / stdExp < 0.001);
    }
}

TEST(Math, NaturalLog)
{
    for (double i = 0.01; i <= 100.0; i += 0.01)
        EXPECT_NEAR(log(i), Math::NaturalLog(i), 0.00001);
}