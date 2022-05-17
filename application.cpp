#include <iostream>
#include <memory>

#include "DataStructures/computation_graph.hpp"

using namespace DataStructure;

int main(void)
{
    ComputationGraph<float> graph;
    auto x = graph.CreateVariableNode(12, "x");
    3 * (x^3) + 5 * (x^2) - 12 * x + 37;
    std::cout << "f(x) = 3x^3 + 5x^2 - 12x + 37 = " << graph.Forward() << std::endl;
    std::cout << "x = " << x.Forward() << std::endl;
    graph.Backward();
    std::cout << "df(x)/dx = " << x.Backward()[0] << std::endl;
    return 0;
}