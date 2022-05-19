namespace DataStructure
{
    template <class T>
    ComputationGraph<T>::ComputationGraph() : nodes(), funcNodes() {}

    template <class T>
    ComputationGraph<T>::~ComputationGraph()
    {
        for (std::size_t i = 0; i < nodes.Size(); i++)
            delete nodes[i];
        nodes.Clear();
    }

    template <class T>
    T ComputationGraph<T>::Forward()
    {
        if (funcNodes.IsEmpty())
            return 0;
        return funcNodes[funcNodes.Size() - 1]->Forward();
    }

    template <class T>
    void ComputationGraph<T>::Backward()
    {
        if (funcNodes.IsEmpty())
            return;
        const std::size_t nNodes = funcNodes.Size();
        funcNodes[nNodes - 1]->gradient = 1;

        std::size_t i = nNodes - 1;
        while (true)
        {
            ComputationGraphNode *input1 = funcNodes[i]->firstInput;
            ComputationGraphNode *input2 = funcNodes[i]->secondInput;
            auto gradients = funcNodes[i]->Backward();
            input1->gradient += gradients[0] * funcNodes[i]->gradient;
            input2->gradient += gradients[1] * funcNodes[i]->gradient;
            if (i == 0)
                break;
            i--;
        }
    }

    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraph<T>::CreateVariableNode(const T &value, const std::string &name)
    {
        VariableNode *newNode = new VariableNode(value, name);
        auto nodeIndex = nodes.Size();
        nodes.Append(newNode);
        return ComputationGraphNodeHandler<T>(this, nodeIndex);
    }

    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraph<T>::CreateFunctionNode(
        const ComputationGraphNodeHandler<T> &inputNodeHandler1,
        const ComputationGraphNodeHandler<T> &inputNodeHandler2,
        const GraphOperation &operation,
        const std::string &name)
    {
        FunctionNode *newNode;
        std::size_t input1Index = inputNodeHandler1.index;
        std::size_t input2Index = inputNodeHandler2.index;
        std::size_t newNodeIndex = nodes.Size();
        switch (operation)
        {
        case GraphOperation::Addition:
            newNode = new AddNode(nodes[input1Index], nodes[input2Index], name);
            break;
        case GraphOperation::Subtraction:
            newNode = new MinusNode(nodes[input1Index], nodes[input2Index], name);
            break;
        case GraphOperation::Multiplication:
            newNode = new MultiplyNode(nodes[input1Index], nodes[input2Index], name);
            break;
        case GraphOperation::Division:
            newNode = new DivideNode(nodes[input1Index], nodes[input2Index], name);
            break;
        case GraphOperation::Power:
            newNode = new PowerNode(nodes[input1Index], nodes[input2Index], name);
            break;
        default:
            throw Exceptions::InvalidArgument("ComputationGraph: Operation not recognized.");
        }
        nodes.Append(newNode);
        funcNodes.Append(newNode);
        return ComputationGraphNodeHandler<T>(this, newNodeIndex);
    }

    template <class T>
    std::string ComputationGraph<T>::ToString() const
    {
        std::stringstream ss;
        ss << "ComputationGraph {\n";
        for (std::size_t i = 0; i < nodes.Size(); i++)
            ss << nodes[i]->ToString() << "\n";
        ss << "}";
        return ss.str();
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &stream, const ComputationGraph<T> &graph)
    {
        stream << graph.ToString();
        return stream;
    }

    template <class T>
    ComputationGraph<T>::ComputationGraphNode::ComputationGraphNode(std::string nodeName)
        : name(nodeName), valuated(false), value(0), gradient(0) {}

    template <class T>
    std::string ComputationGraph<T>::ComputationGraphNode::GetName() const { return name; }

    template <class T>
    std::string ComputationGraph<T>::ComputationGraphNode::ToString() const
    {
        std::stringstream ss;
        ss << name << " "
           << "{\n";
        ss << "  value: " << this->value << ",\n";
        ss << "  gradient: " << this->gradient << "\n}";
        return ss.str();
    }

    template <class T>
    ComputationGraph<T>::VariableNode::VariableNode(T value, std::string nodeName) : ComputationGraphNode(nodeName)
    {
        this->valuated = true;
        this->value = value;
    }

    template <class T>
    T ComputationGraph<T>::VariableNode::Forward() { return this->value; }

    template <class T>
    Tuple<T> ComputationGraph<T>::VariableNode::Backward() { return Tuple<T>({this->gradient}); }

    template <class T>
    ComputationGraph<T>::FunctionNode::FunctionNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName)
        : ComputationGraphNode(nodeName), firstInput(input1), secondInput(input2) {}

    template <class T>
    ComputationGraph<T>::AddNode::AddNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::AddNode::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = this->firstInput->Forward() + this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ComputationGraph<T>::AddNode::Backward() { return Tuple<T>({1, 1}); }

    template <class T>
    ComputationGraph<T>::MinusNode::MinusNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::MinusNode::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = this->firstInput->Forward() - this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ComputationGraph<T>::MinusNode::Backward() { return Tuple<T>({1, -1}); }

    template <class T>
    ComputationGraph<T>::MultiplyNode::MultiplyNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::MultiplyNode::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = this->firstInput->Forward() * this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ComputationGraph<T>::MultiplyNode::Backward()
    {
        return Tuple<T>({this->secondInput->Forward(), this->firstInput->Forward()});
    }

    template <class T>
    ComputationGraph<T>::DivideNode::DivideNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::DivideNode::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = this->firstInput->Forward() / this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ComputationGraph<T>::DivideNode::Backward()
    {
        const T firstInputValue = this->firstInput->Forward();
        const T secondInputValue = this->secondInput->Forward();
        return Tuple<T>({1 / secondInputValue,
                         -firstInputValue / (secondInputValue * secondInputValue)});
    }

    template <class T>
    ComputationGraph<T>::PowerNode::PowerNode(ComputationGraphNode *input1, ComputationGraphNode *input2, std::string nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::PowerNode::Forward()
    {
        if (this->valuated)
            return this->value;
        this->valuated = true;
        return this->value = Math::Power(this->firstInput->Forward(), this->secondInput->Forward());
    }

    template <class T>
    Tuple<T> ComputationGraph<T>::PowerNode::Backward()
    {
        const T firstInputValue = this->firstInput->Forward();
        const T secondInputValue = this->secondInput->Forward();
        const T powered = Math::Power(firstInputValue, secondInputValue - 1);
        return Tuple<T>({secondInputValue * powered,
                         powered * firstInputValue * Math::Log(firstInputValue)});
    }

    template <class T>
    ComputationGraphNodeHandler<T>::ComputationGraphNodeHandler(ComputationGraph<T> *ownerGraph, std::size_t nodeIndex)
        : index(nodeIndex), graph(ownerGraph) {}

    template <class T>
    T ComputationGraphNodeHandler<T>::Forward() const
    {
        return graph->nodes[index]->Forward();
    }

    template <class T>
    Tuple<T> ComputationGraphNodeHandler<T>::Backward() const
    {
        return graph->nodes[index]->Backward();
    }

    template <class T>
    std::string ComputationGraphNodeHandler<T>::GetNodeName() const
    {
        return graph->nodes[index]->GetName();
    }

    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraphNodeHandler<T>::operator+(const ComputationGraphNodeHandler &other) const
    {
        if (this->graph != other.graph)
            throw Exceptions::InvalidArgument(
                "ComputationGraphNodeHandler: "
                "Cannot perform operations on nodes from different CompurationGraphs.");
        std::stringstream ss;
        ss << "AddNode(" << GetNodeName() << ", " << other.GetNodeName() << ")";
        return this->graph->CreateFunctionNode(
            *this, other,
            GraphOperation::Addition,
            ss.str());
    }
    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraphNodeHandler<T>::operator-(const ComputationGraphNodeHandler &other) const
    {
        if (this->graph != other.graph)
            throw Exceptions::InvalidArgument(
                "ComputationGraphNodeHandler: "
                "Cannot perform operations on nodes from different CompurationGraphs.");
        std::stringstream ss;
        ss << "MinusNode(" << GetNodeName() << ", " << other.GetNodeName() << ")";
        return this->graph->CreateFunctionNode(
            *this, other,
            GraphOperation::Subtraction,
            ss.str());
    }

    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraphNodeHandler<T>::operator*(const ComputationGraphNodeHandler &other) const
    {
        if (this->graph != other.graph)
            throw Exceptions::InvalidArgument(
                "ComputationGraphNodeHandler: "
                "Cannot perform operations on nodes from different CompurationGraphs.");
        std::stringstream ss;
        ss << "MultiplyNode(" << GetNodeName() << ", " << other.GetNodeName() << ")";
        return this->graph->CreateFunctionNode(
            *this, other,
            GraphOperation::Multiplication,
            ss.str());
    }

    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraphNodeHandler<T>::operator/(const ComputationGraphNodeHandler &other) const
    {
        if (this->graph != other.graph)
            throw Exceptions::InvalidArgument(
                "ComputationGraphNodeHandler: "
                "Cannot perform operations on nodes from different CompurationGraphs.");
        std::stringstream ss;
        ss << "DivideNode(" << GetNodeName() << ", " << other.GetNodeName() << ")";
        return this->graph->CreateFunctionNode(
            *this, other,
            GraphOperation::Division,
            ss.str());
    }

    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraphNodeHandler<T>::operator^(const ComputationGraphNodeHandler &other) const
    {
        if (this->graph != other.graph)
            throw Exceptions::InvalidArgument(
                "ComputationGraphNodeHandler: "
                "Cannot perform operations on nodes from different CompurationGraphs.");
        std::stringstream ss;
        ss << "PowerNode(" << GetNodeName() << ", " << other.GetNodeName() << ")";
        return this->graph->CreateFunctionNode(
            *this, other,
            GraphOperation::Power,
            ss.str());
    }
}