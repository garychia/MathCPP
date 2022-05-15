#ifndef COMPUTATION_GRAPH_HPP
#define COMPUTATION_GRAPH_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <memory>

#include "list.hpp"
#include "tuple.hpp"
#include "../Exceptions/exceptions.hpp"

namespace DataStructure
{
    template <class T>
    class ComputationGraphNode
    {
    protected:
        bool valuated;
        T value;
        T gradient;

    public:
        ComputationGraphNode();

        virtual T Forward() = 0;

        virtual Tuple<T> Backward() = 0;

        template <class GraphType>
        friend class ComputationGraph;
    };

    template <class T>
    class VariableNode : public ComputationGraphNode<T>
    {
    public:
        VariableNode(T value);

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class FunctionNode : public ComputationGraphNode<T>
    {
    protected:
        using InputNode = std::weak_ptr<ComputationGraphNode<T>>;

    protected:
        Tuple<InputNode> inputs;

    public:
        FunctionNode(InputNode input1, InputNode input2);

        virtual T Forward() = 0;

        virtual Tuple<T> Backward() = 0;

        template <class GraphType>
        friend class ComputationGraph;
    };

    template <class T>
    class AddNode : public FunctionNode<T>
    {
    public:
        AddNode(InputNode input1, InputNode input2);

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class ComputationGraph
    {
    private:
        List<std::shared_ptr<FunctionNode<T>>> nodes;

    public:
        ComputationGraph();

        void AddComputation(const std::shared_ptr<FunctionNode<T>> &computationNode);

        T Forward();

        void Backward();
    };
} // namespace DataStructures

#include "computation_graph.tpp"

#endif