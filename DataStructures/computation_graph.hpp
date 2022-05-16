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
#include "../Algorithms/math.hpp"

#define POINTER_REFERENCE_SIGNATURE(func_name, return_type, type1, arg1, type2, arg2, ...) \
    return_type func_name(type1 *arg1, type2 *arg2, __VA_ARGS__); \
    return_type func_name(type1 &arg1, type2 &arg2, __VA_ARGS__); \
    return_type func_name(type1 *arg1, type2 &arg2, __VA_ARGS__); \
    return_type func_name(type1 &arg1, type2 *arg2, __VA_ARGS__); \

namespace DataStructure
{
    enum class GraphOperation
    {
        Addition,
        Subtraction,
        Multiplication,
        Division,
        Power
    };

    template <class T>
    class ComputationGraph;

    template <class T>
    class FunctionNode;

    template <class T>
    class ComputationGraphNode
    {
    protected:
        std::string name;
        bool valuated;
        T value;
        T gradient;
        ComputationGraph<T> *compGraph;

    public:
        ComputationGraphNode(ComputationGraph<T> *graph, std::string nodeName = "ComputationGraph");

        virtual ~ComputationGraphNode() = default;

        virtual T Forward() = 0;

        virtual Tuple<T> Backward() = 0;

        virtual std::string ToString() const;

        POINTER_REFERENCE_SIGNATURE(
            CombineNodes,
            static FunctionNode<T> *,
            ComputationGraphNode<T>, node1,
            ComputationGraphNode<T>, node2,
            const GraphOperation &operation)

        template <class GraphType>
        friend class ComputationGraph;

        friend std::ostream;
    };

    template <class T>
    class VariableNode : public ComputationGraphNode<T>
    {
    public:
        VariableNode(ComputationGraph<T> *graph, T value, std::string nodeName = "VariableNode");

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
        FunctionNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName = "FunctionNode");

        virtual T Forward() = 0;

        virtual Tuple<T> Backward() = 0;

        template <class GraphType>
        friend class ComputationGraph;
    };

    template <class T>
    class AddNode : public FunctionNode<T>
    {
    public:
        AddNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName = "AddNode");

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class MinusNode : public FunctionNode<T>
    {
    public:
        MinusNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName = "MinusNode");

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class MultiplyNode : public FunctionNode<T>
    {
    public:
        MultiplyNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName = "MultiplyNode");

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class DivideNode : public FunctionNode<T>
    {
    public:
        DivideNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName = "DivideNode");

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class PowerNode : public FunctionNode<T>
    {
    public:
        PowerNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName = "PowerNode");

        virtual T Forward() override;

        virtual Tuple<T> Backward() override;
    };

    template <class T>
    class ComputationGraph
    {
    private:
        List<FunctionNode<T> *> nodes;
        List<ComputationGraphNode<T> *> tempNodes;

    public:
        ComputationGraph();

        ~ComputationGraph();

        void AddComputation(FunctionNode<T> *computationNode);

        T Forward();

        void Backward();

        template <class OtherType>
        friend class ComputationGraphNode;
    };
} // namespace DataStructures

#include "computation_graph.tpp"

#endif