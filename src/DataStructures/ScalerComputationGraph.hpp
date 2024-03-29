#ifndef SCALERCOMPUTATIONGRAPH_HPP
#define SCALERCOMPUTATIONGRAPH_HPP

#include "ComputationGraph.hpp"

namespace DataStructures
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

        class FunctionNode : public ScalerComputationGraphNode
        {
        protected:
            class ComputationGraph<T>::ComputationGraphNode *firstInput;
            class ComputationGraph<T>::ComputationGraphNode *secondInput;

        public:
            FunctionNode(class ComputationGraph<T>::ComputationGraphNode *input1, class ComputationGraph<T>::ComputationGraphNode *input2, const std::string &nodeName = "FunctionNode");

            typename ComputationGraph<T>::ComputationGraphNode *GetFirstInput() const;

            typename ComputationGraph<T>::ComputationGraphNode *GetSecondInput() const;
        };

        class AddNode : public FunctionNode
        {
        public:
            AddNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "AddNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class MinusNode : public FunctionNode
        {
        public:
            MinusNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "MinusNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class MultiplyNode : public FunctionNode
        {
        public:
            MultiplyNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "ScalerMultiplyNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class DivideNode : public FunctionNode
        {
        public:
            DivideNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "DivideNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        class PowerNode : public FunctionNode
        {
        public:
            PowerNode(class ComputationGraph<T>::ComputationGraphNode* input1, class ComputationGraph<T>::ComputationGraphNode* input2, const std::string& nodeName = "PowerNode");

            T Forward() override;

            Tuple<T> Backward() override;
        };

        List<FunctionNode *> funcNodes;

    public:
        ScalerComputationGraph();

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