#include <iostream>
#include <memory>

#include "DataStructures/computation_graph.hpp"

using namespace DataStructure;

int main(void)
{
    ComputationGraph<float> graph;
    auto x = graph.CreateVariableNode(12, "x");
    auto y = graph.CreateVariableNode(12, "y");
    auto result = 3 * (x^3) + 5 * (x^2) - 12 * x + 37 * y;
    
    std::cout << "f(x, y) = 3x^3 + 5x^2 - 12x + 37 * y = " << graph.Forward() << std::endl;
    std::cout << "Graph before Backpropagation:" << std::endl;
    std::cout << graph << std::endl;
    graph.Backward();
    std::cout << "Graph after Backpropagation:" << std::endl;
    std::cout << graph << std::endl;
    return 0;
}