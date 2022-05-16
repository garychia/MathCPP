namespace DataStructure
{
    template <class T>
    ComputationGraphNode<T>::ComputationGraphNode(ComputationGraph<T> *graph, std::string nodeName)
        : name(nodeName), valuated(false), value(), gradient(), compGraph(graph) {}

    template <class T>
    std::string ComputationGraphNode<T>::ToString() const
    {
        std::stringstream ss;
        ss << name << " "
           << "{\n";
        ss << "  value: " << this->value << ",\n";
        ss << "  gradient: " << this->gradient << "\n}";
        return ss.str();
    }

    template <class T>
    FunctionNode<T> *ComputationGraphNode<T>::CombineNodes(
        ComputationGraphNode<T> *node1,
        ComputationGraphNode<T> *node2,
        const GraphOperation &operation)
    {
        ComputationGraph<T> *graph = node1->compGraph;
        FunctionNode<T> *newNode;
        switch (operation)
        {
        case GraphOperation::Addition:
            newNode = new AddNode<T>(graph, node1, node2);
            break;
        case GraphOperation::Subtraction:
            newNode = new MinusNode<T>(graph, node1, node2);
            break;
        case GraphOperation::Multiplication:
            newNode = new MultiplyNode<T>(graph, node1, node2);
            break;
        case GraphOperation::Division:
            newNode = new DivideNode<T>(graph, node1, node2);
            break;
        case GraphOperation::Power:
            newNode = new PowerNode<T>(graph, node1, node2);
            break;
        default:
            return nullptr;
        }
        graph->tempNodes.Append(newNode);
        return newNode;
    }

    template <class T>
    FunctionNode<T> *ComputationGraphNode<T>::CombineNodes(
        ComputationGraphNode<T> &node1,
        ComputationGraphNode<T> &node2,
        const GraphOperation &operation)
    {
        return ComputationGraphNode<T>::CombineNodes(&node1, &node2, operation);
    }

    template <class T>
    FunctionNode<T> *ComputationGraphNode<T>::CombineNodes(
        ComputationGraphNode<T> *node1,
        ComputationGraphNode<T> &node2,
        const GraphOperation &operation)
    {
        return ComputationGraphNode<T>::CombineNodes(node1, &node2, operation);
    }

    template <class T>
    FunctionNode<T> *ComputationGraphNode<T>::CombineNodes(
        ComputationGraphNode<T> &node1,
        ComputationGraphNode<T> *node2,
        const GraphOperation &operation)
    {
        return ComputationGraphNode<T>::CombineNodes(&node1, node2, operation);
    }

    template <class T>
    std::ostream &operator<<(std::ostream &stream, const ComputationGraphNode<T> &node)
    {
        stream << node.ToString();
        return stream;
    }

    template <class T>
    VariableNode<T>::VariableNode(ComputationGraph<T> *graph, T value, std::string nodeName) : ComputationGraphNode<T>(graph, nodeName)
    {
        this->valuated = false;
        this->value = value;
    }

    template <class T>
    T VariableNode<T>::Forward() { return this->value; }

    template <class T>
    Tuple<T> VariableNode<T>::Backward() { return Tuple<T>({this->gradient}); }

    template <class T>
    FunctionNode<T>::FunctionNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName)
        : ComputationGraphNode<T>(graph, nodeName), firstInput(input1), secondInput(input2) {}

    template <class T>
    AddNode<T>::AddNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName)
        : FunctionNode<T>(graph, input1, input2, nodeName) {}

    template <class T>
    T AddNode<T>::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = this->firstInput->Forward() + this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> AddNode<T>::Backward() { return Tuple<T>({1, 1}); }

    template <class T>
    MinusNode<T>::MinusNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName)
        : FunctionNode<T>(graph, input1, input2, nodeName) {}

    template <class T>
    T MinusNode<T>::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = this->firstInput->Forward() - this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> MinusNode<T>::Backward() { return Tuple<T>({1, -1}); }

    template <class T>
    MultiplyNode<T>::MultiplyNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName)
        : FunctionNode<T>(graph, input1, input2, nodeName) {}

    template <class T>
    T MultiplyNode<T>::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = this->firstInput->Forward() * this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> MultiplyNode<T>::Backward() { return Tuple<T>({this->secondInput->Forward(), this->firstInput->Forward()}); }

    template <class T>
    DivideNode<T>::DivideNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName)
        : FunctionNode<T>(graph, input1, input2, nodeName) {}

    template <class T>
    T DivideNode<T>::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = this->firstInput->Forward() / this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> DivideNode<T>::Backward()
    {
        const T firstInputValue = this->firstInput->Forward();
        const T secondInputValue = this->secondInput->Forward();
        return Tuple<T>({1 / secondInputValue,
                         -firstInputValue / (secondInputValue * secondInputValue)});
    }

    template <class T>
    PowerNode<T>::PowerNode(ComputationGraph<T> *graph, ComputationGraphNode<T> *input1, ComputationGraphNode<T> *input2, std::string nodeName)
        : FunctionNode<T>(graph, input1, input2, nodeName) {}

    template <class T>
    T PowerNode<T>::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = Math::Power(this->firstInput->Forward(), this->secondInput->Forward());
    }

    template <class T>
    Tuple<T> PowerNode<T>::Backward()
    {
        const T firstInputValue = this->firstInput->Forward();
        const T secondInputValue = this->secondInput->Forward();
        const T powered = Math::Power(firstInputValue, secondInputValue - 1);
        return Tuple<T>({secondInputValue * powered,
                         powered * firstInputValue * Math::Log(firstInputValue)});
    }

    template <class T>
    ComputationGraph<T>::ComputationGraph() : nodes(), tempNodes() {}

    template <class T>
    ComputationGraph<T>::~ComputationGraph()
    {
        for (std::size_t i = 0; i < tempNodes.Size(); i++)
            delete tempNodes[i];
        tempNodes.Clear();
    }

    template <class T>
    void ComputationGraph<T>::AddComputation(FunctionNode<T> *computationNode)
    {
        List<FunctionNode<T> *> nodesFound({computationNode});
        while (!nodesFound.IsEmpty())
        {
            FunctionNode<T> *currentNode = nodesFound.PopFront();
            nodes.Prepend(currentNode);
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
            input1->gradient += gradients[0] * nodes[i]->gradient;
            input2->gradient += gradients[1] * nodes[i]->gradient;
            if (i == 0)
                break;
            i--;
        }
    }
}