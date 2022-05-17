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
    class ComputationGraphNodeHandler;

    template <class T>
    class ComputationGraph
    {
    private:
        class ComputationGraphNode
        {
        protected:
            std::string name;
            bool valuated;
            T value;
            T gradient;

        public:
            ComputationGraphNode(std::string nodeName = "ComputationGraph");

            virtual ~ComputationGraphNode() = default;

            std::string GetName() const;

            virtual T Forward() = 0;

            virtual Tuple<T> Backward() = 0;

            virtual std::string ToString() const;

            friend class ComputationGraph<T>;
        };

        class VariableNode : public ComputationGraphNode
        {
        public:
            VariableNode(T value, std::string nodeName = "VariableNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        class FunctionNode : public ComputationGraphNode
        {
        protected:
            ComputationGraphNode *firstInput;
            ComputationGraphNode *secondInput;

        public:
            FunctionNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName = "FunctionNode");

            virtual T Forward() = 0;

            virtual Tuple<T> Backward() = 0;

            friend class ComputationGraph<T>;
        };

        class AddNode : public FunctionNode
        {
        public:
            AddNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName = "AddNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        class MinusNode : public FunctionNode
        {
        public:
            MinusNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName = "MinusNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        class MultiplyNode : public FunctionNode
        {
        public:
            MultiplyNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName = "MultiplyNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        class DivideNode : public FunctionNode
        {
        public:
            DivideNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName = "DivideNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        class PowerNode : public FunctionNode
        {
        public:
            PowerNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName = "PowerNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        List<ComputationGraphNode *> nodes;

        List<FunctionNode *> funcNodes;

    public:
        ComputationGraph();

        ~ComputationGraph();

        ComputationGraphNodeHandler<T> CreateVariableNode(const T &value, const std::string &name = "VariableNode");

        ComputationGraphNodeHandler<T> CreateFunctionNode(
            const ComputationGraphNodeHandler<T> &inputNodeHandler1,
            const ComputationGraphNodeHandler<T> &inputNodeHandler2,
            const GraphOperation &operation,
            const std::string &name = "FunctionNode");

        T Forward();

        void Backward();

        std::string ToString() const;

        friend T ComputationGraphNodeHandler<T>::Forward() const;

        friend Tuple<T> ComputationGraphNodeHandler<T>::Backward() const;

        friend std::string ComputationGraphNodeHandler<T>::GetNodeName() const;

        friend std::ostream;
    };

    template <class T>
    class ComputationGraphNodeHandler
    {
    private:
        std::size_t index;

        ComputationGraph<T> *graph;

    public:
        ComputationGraphNodeHandler(ComputationGraph<T> *ownerGraph, std::size_t nodeIndex);

        T Forward() const;

        Tuple<T> Backward() const;

        std::string GetNodeName() const;

        ComputationGraphNodeHandler<T> operator+(const ComputationGraphNodeHandler &other) const;
        ComputationGraphNodeHandler<T> operator-(const ComputationGraphNodeHandler &other) const;
        ComputationGraphNodeHandler<T> operator*(const ComputationGraphNodeHandler &other) const;
        ComputationGraphNodeHandler<T> operator/(const ComputationGraphNodeHandler &other) const;
        ComputationGraphNodeHandler<T> operator^(const ComputationGraphNodeHandler &other) const;

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator+(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "AddNode(" << scaler << ", " << handler.GetNodeName() << ")";
            return graph->CreateFunctionNode(scalerVariable, handler, GraphOperation::Addition, ss.str());
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator+(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "AddNode(" << handler.GetNodeName() << ", " << scaler << ")";
            return graph->CreateFunctionNode(handler, scalerVariable, GraphOperation::Addition, ss.str());
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator-(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "MinusNode(" << scaler << ", " << handler.GetNodeName() << ")";
            return graph->CreateFunctionNode(scalerVariable, handler, GraphOperation::Subtraction, ss.str());
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator-(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "MinusNode(" << handler.GetNodeName() << ", " << scaler << ")";
            return graph->CreateFunctionNode(handler, scalerVariable, GraphOperation::Subtraction, ss.str());
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator*(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "MultiplyNode(" << scaler << ", " << handler.GetNodeName() << ")";
            return graph->CreateFunctionNode(scalerVariable, handler, GraphOperation::Multiplication, ss.str());
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator*(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "MultiplyNode(" << handler.GetNodeName() << ", " << scaler << ")";
            return graph->CreateFunctionNode(handler, scalerVariable, GraphOperation::Multiplication, ss.str());
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator/(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "DivideNode(" << scaler << ", " << handler.GetNodeName() << ")";
            return graph->CreateFunctionNode(scalerVariable, handler, GraphOperation::Division, ss.str());
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator/(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "DivideNode(" << handler.GetNodeName() << ", " << scaler << ")";
            return graph->CreateFunctionNode(handler, scalerVariable, GraphOperation::Division, ss.str());
        }
        
        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator^(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "PowerNode(" << scaler << ", " << handler.GetNodeName() << ")";
            return graph->CreateFunctionNode(scalerVariable, handler, GraphOperation::Power, ss.str());
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator^(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateVariableNode(scaler);
            std::stringstream ss;
            ss << "PowerNode(" << handler.GetNodeName() << ", " << scaler << ")";
            return graph->CreateFunctionNode(handler, scalerVariable, GraphOperation::Power, ss.str());
        }

        friend class ComputationGraph<T>;
    };

} // namespace DataStructures

#include "computation_graph.tpp"

#endif