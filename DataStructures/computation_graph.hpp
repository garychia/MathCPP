#ifndef COMPUTATION_GRAPH_HPP
#define COMPUTATION_GRAPH_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <string>
#include <sstream>
#include <ostream>

#include "list.hpp"
#include "tuple.hpp"
#include "../Exceptions/exceptions.hpp"

namespace DataStructure
{
    template <class T>
    class ComputationGraphNode
    {
    protected:
        std::string name;
        bool valuated;
        T value;
        T gradient;

    public:
        ComputationGraphNode(std::string nodeName = "ComputationGraph");

        virtual T Forward() = 0;

        virtual Tuple<T> Backward() = 0;

        virtual std::string ToString() const;

        template <class GraphType>
        friend class ComputationGraph;

        friend std::ostream;
    };

    template <class T>
    class VariableNode : public ComputationGraphNode<T>
    {
    public:
        VariableNode(T value, std::string nodeName = "VariableNode");

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class FunctionNode : public ComputationGraphNode<T>
    {
    protected:
        ComputationGraphNode<T> *firstInput;
        ComputationGraphNode<T> *secondInput;

    public:
        FunctionNode(ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName = "FunctionNode");

        virtual T Forward() = 0;

        virtual Tuple<T> Backward() = 0;

        template <class GraphType>
        friend class ComputationGraph;
    };

    template <class T>
    class AddNode : public FunctionNode<T>
    {
    public:
        AddNode(ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName = "AddNode");

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class MinusNode : public FunctionNode<T>
    {
    public:
        MinusNode(ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName = "MinusNode");

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class ComputationGraph
    {
    private:
        List<FunctionNode<T> *> nodes;

    public:
        ComputationGraph();

        void AddComputation(FunctionNode<T> *computationNode);

        T Forward();

        void Backward();
    };
} // namespace DataStructures

#include "computation_graph.tpp"

#endif