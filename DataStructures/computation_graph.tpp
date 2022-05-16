namespace DataStructure
{
    template <class T>
    ComputationGraphNode<T>::ComputationGraphNode() : valuated(false), value(), gradient() {}

    template <class T>
    std::ostream &operator<<(std::ostream &stream, const ComputationGraphNode<T> &node)
    {
        stream << node.ToString();
        return stream;
    }

    template <class T>
    VariableNode<T>::VariableNode(T value)
    {
        this->valuated = false;
        this->value = value;
    }

    template <class T>
    T VariableNode<T>::Forward() { return this->value; }

    template <class T>
    Tuple<T> VariableNode<T>::Backward() { return Tuple<T>({1}); }

    template <class T>
    std::string VariableNode<T>::ToString() const
    {
        std::stringstream ss;
        ss << "VariableNode {";
        ss << "value: " << this->value << ", ";
        ss << "gradient: " << this->gradient << "}";
        return ss.str();
    }

    template <class T>
    FunctionNode<T>::FunctionNode(ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2)
        : ComputationGraphNode<T>(), firstInput(input1), secondInput(input2) {}

    template <class T>
    AddNode<T>::AddNode(ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2) : FunctionNode<T>(input1, input2) {}

    template <class T>
    T AddNode<T>::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = this->firstInput->Forward() + this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> AddNode<T>::Backward()
    {
        return Tuple<T>({1, 1});
    }

    template <class T>
    std::string AddNode<T>::ToString() const
    {
        std::stringstream ss;
        ss << "AddNode {";
        ss << "value: ";
        if (this->valuated)
            ss << this->value;
        else
            ss << "NOT VALUATED";
        ss << ", ";
        ss << "gradient: " << this->gradient << "}";
        return ss.str();
    }

    template <class T>
    ComputationGraph<T>::ComputationGraph() : nodes() {}

    template <class T>
    void ComputationGraph<T>::AddComputation(FunctionNode<T> *computationNode)
    {
        List<FunctionNode<T> *> nodesFound({computationNode});
        while (!nodesFound.IsEmpty())
        {
            FunctionNode<T> *currentNode = nodesFound.PopFront();
            nodes.Append(currentNode);
            ComputationGraphNode<T> *input1 = currentNode->firstInput;
            ComputationGraphNode<T> *input2 = currentNode->secondInput;
            if (auto funcNode = dynamic_cast<FunctionNode<T> *>(input1))
                nodesFound.Append(funcNode);
            if (auto funcNode = dynamic_cast<FunctionNode<T> *>(input2))
                nodesFound.Append(funcNode);
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
            ComputationGraphNode<T> *input1 = nodes[i]->firstInput;
            ComputationGraphNode<T> *input2 = nodes[i]->secondInput;
            auto gradients = nodes[i]->Backward();
            input1->gradient = gradients[0] * nodes[i]->gradient;
            input2->gradient = gradients[1] * nodes[i]->gradient;
            if (i == 0)
                break;
            i--;
        }
    }
}