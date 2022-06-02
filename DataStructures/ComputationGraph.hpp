#ifndef COMPUTATION_GRAPH_HPP
#define COMPUTATION_GRAPH_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <string>
#include <sstream>
#include <ostream>

#include "List.hpp"
#include "Tuple.hpp"
#include "Exceptions.hpp"
#include "Math.hpp"

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

        List<ComputationGraphNode *> nodes;

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
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return scalerVariable + handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator+(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return handler + scalerVariable;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator-(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return scalerVariable - handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator-(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return handler - scalerVariable;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator*(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return scalerVariable * handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator*(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return handler * scalerVariable;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator/(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return scalerVariable / handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator/(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return handler / scalerVariable;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator^(const OtherType &scaler, const ComputationGraphNodeHandler<T> &handler)
        {
            ComputationGraph<T> *graph = handler.graph;
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return scalerVariable ^ handler;
        }

        template <class OtherType>
        friend ComputationGraphNodeHandler<T> operator^(const ComputationGraphNodeHandler<T> &handler, const OtherType &scaler)
        {
            ComputationGraph<T> *graph = handler.graph;
            std::stringstream ss;
            ss << "Constant(" << scaler << ")";
            auto scalerVariable = graph->CreateConstantNode(scaler, ss.str());
            return handler ^ scalerVariable;
        }

        friend class ComputationGraph<T>;

        template <class U>
        friend class ScalerComputationGraph;
        
        template <class U>
        friend class MatrixComputationGraph;
    };

} // namespace DataStructures

#include "ComputationGraph.tpp"

#endif