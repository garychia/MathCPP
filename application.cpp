#include <iostream>
#include <memory>

#include "DataStructures/computation_graph.hpp"

using namespace DataStructure;

int main(void)
{
    ComputationGraph<float> graph;
    VariableNode<float> x(3);
    VariableNode<float> x2Expo(2);
    VariableNode<float> xC(3);
    VariableNode<float> c(85);
    PowerNode<float> x2Term(&x, &x2Expo);
    MultiplyNode<float> xTerm(&x, &xC);
    AddNode<float> add1(&x2Term, &xTerm);
    AddNode<float> add2(&add1, &c);
    graph.AddComputation(&add2);
    graph.Backward();
    std::cout << "f(x) = x ^ 2 + 3x + 85" << std::endl;
    std::cout << "x = " << x.Forward() << std::endl;
    std::cout << "f(x) = " << graph.Forward() << std::endl;
    std::cout << "dx = " << x.Backward() << std::endl;
    return 0;
}