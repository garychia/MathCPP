#include <iostream>
#include <memory>

#include "DataStructures/computation_graph.hpp"

using namespace DataStructure;

int main(void)
{
    ComputationGraph<float> graph;
    VariableNode<float> a(&graph, 5);
    VariableNode<float> b(&graph, 13);
    VariableNode<float> c(&graph, 23);
    auto expression =
        ComputationGraphNode<float>::CombineNodes(
            ComputationGraphNode<float>::CombineNodes(
                a,
                b,
                GraphOperation::Addition),
            c,
            GraphOperation::Division);
    graph.AddComputation(expression);
    std::cout << "(a + b) / c = " << graph.Forward() << std::endl;
    std::cout << "a = " << a.Forward() << std::endl;
    std::cout << "b = " << b.Forward() << std::endl;
    std::cout << "c = " << c.Forward() << std::endl;
    graph.Backward();
    std::cout << "da = " << a.Backward() << std::endl;
    std::cout << "db = " << b.Backward() << std::endl;
    std::cout << "dc = " << c.Backward() << std::endl;
    return 0;
}