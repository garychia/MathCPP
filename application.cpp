#include <iostream>
#include <memory>

#include "ScalerComputationGraph.hpp"

using namespace DataStructure;

int main(void)
{
    ScalerComputationGraph<float> graph;
    auto x = graph.CreateVariableNode(12, "x");
    auto y = graph.CreateVariableNode(12, "y");
    3 * (x^3) + 5 * (x^2) - 12 * x + 37 * y + x / y;
    std::cout << "f(x, y) = 3x^3 + 5x^2 - 12x + 37 * y + x / y = " << graph.Forward() << std::endl;
    graph.Backward();
    std::cout << "x = " << x.Forward() << std::endl;
    std::cout << "y = " << y.Forward() << std::endl;
    std::cout << "df(x, y)/dx = " << x.Gradient() << std::endl;
    std::cout << "df(x, y)/dy = " << y.Gradient() << std::endl;
    std::cout << "...........................................\n";
    x.SetValue(23);
    y.SetValue(7);
    std::cout << "f(x, y) = 3x^3 + 5x^2 - 12x + 37 * y + x / y = " << graph.Forward() << std::endl;
    graph.Backward();
    std::cout << "x = " << x.Forward() << std::endl;
    std::cout << "y = " << y.Forward() << std::endl;
    std::cout << "df(x, y)/dx = " << x.Gradient() << std::endl;
    std::cout << "df(x, y)/dy = " << y.Gradient() << std::endl;
    return 0;
}