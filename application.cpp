#include <iostream>
#include <memory>

#include "DataStructures/computation_graph.hpp"

using namespace DataStructure;

int main(void)
{
    
    ComputationGraph<float> graph;
    //auto a = std::make_shared<VariableNode<float>>(12);
    //auto b = std::make_shared<VariableNode<float>>(32);
    //std::shared_ptr<FunctionNode<float>> addition = std::make_shared<AddNode<float>>(a, b);
    //graph.AddComputation(addition);
    
    //std::cout << graph.Forward() << std::endl;
    //graph.Backward();
    //std::cout << "a = " << a->Forward() << std::endl;
    //std::cout << "b = " << b->Forward() << std::endl;
    //std::cout << "da = " << a->Backward() << std::endl;
    //std::cout << "db = " << b->Backward() << std::endl;
    
    return 0;
}