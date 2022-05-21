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
    protected:
        class ComputationGraphNode
        {
        protected:
            std::string name;
            bool gradientValuated;
            T gradient;

        public:
            ComputationGraphNode(const std::string &nodeName = "ComputationGraph");

            virtual ~ComputationGraphNode() = default;

            virtual T Forward() = 0;

            virtual Tuple<T> Backward() = 0;

            virtual void Reset() = 0;

            virtual void UpdateGradient(const T &partialGradient) = 0;

            std::string GetName() const;

            T GetGradient() const;

            T GetPartialGradient() const;

            bool IsGradientValuated() const;

            void MarkGradientValuated();

            virtual std::string ToString() const;
        };


        class FunctionNode : public ComputationGraphNode
        {
        protected:
            ComputationGraphNode *firstInput;
            ComputationGraphNode *secondInput;

        public:
            FunctionNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName = "FunctionNode");

            ComputationGraphNode *GetFirstInput() const;

            ComputationGraphNode *GetSecondInput() const;
        };

        List<ComputationGraphNode *> nodes;

        List<FunctionNode *> funcNodes;

        void reset() const;

    public:
        ComputationGraph();

        virtual ~ComputationGraph();

        virtual ComputationGraphNodeHandler<T> CreateConstantNode(const T &value, const std::string &name = "ConstantNode") = 0;

        virtual ComputationGraphNodeHandler<T> CreateVariableNode(const T &value, const std::string &name = "VariableNode") = 0;

        virtual ComputationGraphNodeHandler<T> CreateFunctionNode(
            const ComputationGraphNodeHandler<T> &inputNodeHandler1,
            const ComputationGraphNodeHandler<T> &inputNodeHandler2,
            const GraphOperation &operation,
            const std::string &name) = 0;

        virtual T Forward() = 0;

        virtual void Backward() = 0;

        T GetValue(const ComputationGraphNodeHandler<T> &handler) const;

        virtual void SetValue(const ComputationGraphNodeHandler<T> &handler, const T &newValue) const = 0;

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
            ss << "Constant(" << scaler << ")";
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

        template <class U>
        friend class ScalerComputationGraph;
        
        template <class U>
        friend class MatrixComputationGraph;
    };

    template <class T>
    class ScalerComputationGraph : public ComputationGraph<T>
    {
    private:
        class ScalerComputationGraphNode : public ComputationGraphNode
        {
        public:
            ScalerComputationGraphNode(const std::string& nodeName = "ScalerComputationGraphNode");

            void Reset() override;

            void UpdateGradient(const T& partialGradient) override;
        };

        class ConstantNode : public ScalerComputationGraphNode
        {
        protected:
            T value;

        public:
            ConstantNode(const T& value, const std::string& nodeName = "ConstantNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class VariableNode : public ConstantNode
        {
        public:
            VariableNode(const T& value, const std::string& nodeName = "VariableNode");

            void SetValue(const T& newValue);
        };

        class ScalerFunctionNode : public FunctionNode
        {
        public:
            ScalerFunctionNode(class ComputationGraphNode* input1, class  ComputationGraphNode* input2, const std::string& nodeName = "ScalerFunctionNode");

            void Reset() override;

            void UpdateGradient(const T& partialGradient) override;
        };

        class AddNode : public ScalerFunctionNode
        {
        public:
            AddNode(ComputationGraphNode* input1, ComputationGraphNode* input2, const std::string& nodeName = "AddNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class MinusNode : public ScalerFunctionNode
        {
        public:
            MinusNode(class ComputationGraphNode* input1, class ComputationGraphNode* input2, const std::string& nodeName = "MinusNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class MultiplyNode : public ScalerFunctionNode
        {
        public:
            MultiplyNode(class ComputationGraphNode* input1, class ComputationGraphNode* input2, const std::string& nodeName = "ScalerMultiplyNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class DivideNode : public ScalerFunctionNode
        {
        public:
            DivideNode(class ComputationGraphNode* input1, class ComputationGraphNode* input2, const std::string& nodeName = "DivideNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class PowerNode : public ScalerFunctionNode
        {
        public:
            PowerNode(class ComputationGraphNode* input1, class ComputationGraphNode* input2, const std::string& nodeName = "PowerNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

    public:
        T Forward() override;

        void Backward() override;

        void SetValue(const ComputationGraphNodeHandler<T>& handler, const T& newValue) const override;

        ComputationGraphNodeHandler<T> CreateConstantNode(const T& value, const std::string& name) override;

        ComputationGraphNodeHandler<T> CreateVariableNode(const T& value, const std::string& name) override;

        ComputationGraphNodeHandler<T> CreateFunctionNode(
            const ComputationGraphNodeHandler<T>& inputNodeHandler1,
            const ComputationGraphNodeHandler<T>& inputNodeHandler2,
            const GraphOperation& operation,
            const std::string& name) override;
    };

} // namespace DataStructures

#include "computation_graph.tpp"

#endif