#ifndef SCALERCOMPUTATIONGRAPH_HPP
#define SCALERCOMPUTATIONGRAPH_HPP

#include "ComputationGraph.hpp"

namespace DataStructure
{
    template <class T>
    class ScalerComputationGraph : public ComputationGraph<T>
    {
    private:
        class ScalerComputationGraphNode : public ComputationGraph<T>::ComputationGraphNode
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

        class ScalerFunctionNode : public ComputationGraph<T>::FunctionNode
        {
        public:
            ScalerFunctionNode(class ComputationGraph<T>::ComputationGraphNode* input1, class  ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "ScalerFunctionNode");

            void Reset() override;

            void UpdateGradient(const T& partialGradient) override;
        };

        class AddNode : public ScalerFunctionNode
        {
        public:
            AddNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "AddNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class MinusNode : public ScalerFunctionNode
        {
        public:
            MinusNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "MinusNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class MultiplyNode : public ScalerFunctionNode
        {
        public:
            MultiplyNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "ScalerMultiplyNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class DivideNode : public ScalerFunctionNode
        {
        public:
            DivideNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "DivideNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class PowerNode : public ScalerFunctionNode
        {
        public:
            PowerNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "PowerNode");

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
} // namespace DataStructure

#include "ScalerComputationGraph.tpp"

#endif