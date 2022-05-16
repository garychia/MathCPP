#include <iostream>
#include <memory>

#include "DataStructures/computation_graph.hpp"

using namespace DataStructure;

int main(void)
{
    ComputationGraph<float> graph;
    VariableNode<float> x(&graph, 10);
    VariableNode<float> p1(&graph, 3);
    VariableNode<float> p2(&graph, 2);
    VariableNode<float> p3(&graph, 1);
    VariableNode<float> c1(&graph, 3);
    VariableNode<float> c2(&graph, 5);
    VariableNode<float> c3(&graph, 12);
    VariableNode<float> c4(&graph, 37);
    auto expression =
        ComputationGraphNode<float>::CombineNodes(
            ComputationGraphNode<float>::CombineNodes(
                ComputationGraphNode<float>::CombineNodes(
                    ComputationGraphNode<float>::CombineNodes(
                        ComputationGraphNode<float>::CombineNodes(
                            x,
                            p1,
                            GraphOperation::Power),
                        c1,
                        GraphOperation::Multiplication),
                    ComputationGraphNode<float>::CombineNodes(
                        ComputationGraphNode<float>::CombineNodes(
                            x,
                            p2,
                            GraphOperation::Power),
                        c2,
                        GraphOperation::Multiplication),
                    GraphOperation::Addition),
                ComputationGraphNode<float>::CombineNodes(
                    ComputationGraphNode<float>::CombineNodes(
                        x,
                        p3,
                        GraphOperation::Power),
                    c3,
                    GraphOperation::Multiplication),
                GraphOperation::Subtraction),
            c4,
            GraphOperation::Addition);
    graph.AddComputation(expression);
    std::cout << "f(x) = 3x^3 + 5x^2 - 12x + 37 = " << graph.Forward() << std::endl;
    std::cout << "x = " << x.Forward() << std::endl;
    graph.Backward();
    std::cout << "df(x)/dx = " << x.Backward()[0] << std::endl;
    return 0;
}