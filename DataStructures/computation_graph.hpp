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
            bool gradientValuated;
            T gradient;

        public:
            ComputationGraphNode(const std::string &nodeName = "ComputationGraph");

            virtual ~ComputationGraphNode() = default;

            std::string GetName() const;

            virtual T Forward() = 0;

            virtual Tuple<T> Backward() = 0;

            virtual void Reset();

            virtual void UpdateGradient(T partialGradient);

            virtual void MarkGradientValuated();

            virtual T GetGradient() const;

            virtual T GetPartialGradient() const;

            virtual std::string ToString() const;
        };

        class ConstantNode : public ComputationGraphNode
        {
        protected:
            T value;

        public:
            ConstantNode(const T &value, const std::string &nodeName = "ConstantNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;

            virtual std::string ToString() const override;
        };

        class VariableNode : public ConstantNode
        {
        public:
            VariableNode(const T &value, const std::string &nodeName = "VariableNode");

            void SetValue(const T &newValue);
        };

        class FunctionNode : public ComputationGraphNode
        {
        protected:
            ComputationGraphNode *firstInput;
            ComputationGraphNode *secondInput;

        public:
            FunctionNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName = "FunctionNode");

            virtual T Forward() = 0;

            virtual Tuple<T> Backward() = 0;

            ComputationGraphNode *GetFirstInput() const;

            ComputationGraphNode *GetSecondInput() const;
        };

        class AddNode : public FunctionNode
        {
        public:
            AddNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName = "AddNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        class MinusNode : public FunctionNode
        {
        public:
            MinusNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName = "MinusNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        class MultiplyNode : public FunctionNode
        {
        public:
            MultiplyNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName = "MultiplyNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        class DivideNode : public FunctionNode
        {
        public:
            DivideNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName = "DivideNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        class PowerNode : public FunctionNode
        {
        public:
            PowerNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName = "PowerNode");

            virtual T Forward() override;

            virtual Tuple<T> Backward() override;
        };

        List<ComputationGraphNode *> nodes;

        List<FunctionNode *> funcNodes;

    private:
        void reset() const;

    public:
        ComputationGraph();

        ~ComputationGraph();

        ComputationGraphNodeHandler<T> CreateConstantNode(const T &value, const std::string &name = "ConstantNode");

        ComputationGraphNodeHandler<T> CreateVariableNode(const T &value, const std::string &name = "VariableNode");

        ComputationGraphNodeHandler<T> CreateFunctionNode(
            const ComputationGraphNodeHandler<T> &inputNodeHandler1,
            const ComputationGraphNodeHandler<T> &inputNodeHandler2,
            const GraphOperation &operation,
            const std::string &name = "FunctionNode");

        T Forward();

        void Backward();

        T GetValue(const ComputationGraphNodeHandler<T> &handler) const;

        void SetValue(const ComputationGraphNodeHandler<T> &handler, const T &newValue) const;

        T GetGradient(const ComputationGraphNodeHandler<T> &handler) const;

        std::string GetNodeName(const ComputationGraphNodeHandler<T> &handler) const;

        std::string ToString() const;

        friend std::ostream;
    };

    template <class T>
    class ComputationGraphNodeHandler
    {
    private:
        std::size_t index;

        ComputationGraph<T> *graph;

        bool isVariable;

    public:
        ComputationGraphNodeHandler(ComputationGraph<T> *ownerGraph, std::size_t nodeIndex, bool isVariable = false);

        T Forward() const;

        T Gradient() const;

        std::string GetNodeName() const;

        bool IsVariable() const;

        void SetValue(const T &newValue) const;

        ComputationGraphNodeHandler<T> operator+(const ComputationGraphNodeHandler<T> &other) const;
        ComputationGraphNodeHandler<T> operator-(const ComputationGraphNodeHandler<T> &other) const;
        ComputationGraphNodeHandler<T> operator*(const ComputationGraphNodeHandler<T> &other) const;
        ComputationGraphNodeHandler<T> operator/(const ComputationGraphNodeHandler<T> &other) const;
        ComputationGraphNodeHandler<T> operator^(const ComputationGraphNodeHandler<T> &other) const;

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator+(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return scalerVariable + handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator+(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return handler + scalerVariable;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator-(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return scalerVariable - handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator-(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return handler - scalerVariable;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator*(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return scalerVariable * handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator*(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return handler * scalerVariable;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator/(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return scalerVariable / handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator/(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return handler / scalerVariable;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator^(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return scalerVariable ^ handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator^(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            auto scalerVariable = graph->CreateConstantNode(scaler);
            return handler ^ scalerVariable;
        }

        friend class ComputationGraph<T>;
    };

    template <class MatrixElementType>
    class ComputationGraphNodeHandler<Matrix<MatrixElementType>>;

    template <class MatrixElementType>
    class ComputationGraph<Matrix<MatrixElementType>>;

} // namespace DataStructures

#include "computation_graph.tpp"

#endif