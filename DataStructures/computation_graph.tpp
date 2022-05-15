namespace DataStructure
{
    template <class T>
    ComputationGraphNode<T>::ComputationGraphNode() : valuated(false), value(), gradient() {}

    template <class T>
    VariableNode<T>::VariableNode(T value) : ComputationGraphNode<T>(), valuated(true), value(value), gradient(1) {}

    template <class T>
    T VariableNode<T>::Forward() { return value; }

    template <class T>
    Tuple<T> VariableNode<T>::Backward() { return Tuple<T>({gradient}); }

    template <class T>
    FunctionNode<T>::FunctionNode(InputNode input1, InputNode input2) : ComputationGraphNode<T>(), inputs({input1, input2}) {}

    template <class T>
    AddNode<T>::AddNode(InputNode input1, InputNode input2) : FunctionNode<T>(input1, input2) {}

    template <class T>
    T AddNode<T>::Forward()
    {
        if (valuated)
            return value;
        auto firstInput = inputs[0].lock();
        auto secondInput = inputs[1].lock();
        if (firstInput && secondInput)
        {
            return value = firstInput->Forward() + secondInput->Forward();
            valuated = true;
        }
        else
            throw Exceptions::NodeNotFound(
                "AddNode: A node is missing when performing addition.");
    }

    template <class T>
    Tuple<T> AddNode<T>::Backward()
    {
        return Tuple<T>({1, 1});
    }

    template <class T>
    ComputationGraph<T>::ComputationGraph() : nodes() {}

    template <class T>
    void ComputationGraph<T>::AddComputation(const std::shared_ptr<FunctionNode<T>> &computationNode)
    {
        List<std::shared_ptr<FunctionNode<T>>> nodesFound({computationNode});
        while (!nodesFound.IsEmpty())
        {
            std::shared_ptr<FunctionNode<T>> currentNode = nodesFound.PopFront();
            nodes.Append(currentNode);
            Tuple<std::weak_ptr<ComputationGraphNode<T>>> &inputs = currentNode->inputs;
            for (std::size_t i = 0; i < inputs.Size(); i++)
            {
                if (auto node = std::static_pointer_cast<FunctionNode<T>>(inputs[i].lock()))
                    nodesFound.Append(node);
            }
        }
    }

    template <class T>
    T ComputationGraph<T>::Forward()
    {
        if (nodes.IsEmpty())
            return 0;
        for (std::size_t i = 0; i < nodes.Size(); i++)
            nodes[i]->Forward();
        return nodes[nodes.Size() - 1]->Forward();
    }

    template <class T>
    void ComputationGraph<T>::Backward()
    {
        if (nodes.IsEmpty())
            return;
        const std::size_t nNodes = nodes.Size();
        nodes[nNodes - 1]->gradient = 1;

        std::size_t i = nNodes - 1;
        while (true)
        {
            auto &inputs = nodes[i]->inputs;
            auto gradients = nodes[i]->Backward();
            for (std::size_t j = 0; j < inputs.Size(); j++)
            {
                if (auto node = inputs[j].lock())
                    node->gradient = gradients[j] * nodes[i]->gradient;
                else
                    throw Exceptions::NodeNotFound(
                        "ComputationGraph: A node is missing when performing backpropagation.");
            }
            if (i == 0)
                break;
            i--;
        }
    }
}