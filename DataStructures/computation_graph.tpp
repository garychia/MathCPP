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
    void ComputationGraph<T>::reset() const
    {
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nodes.Size(); i++)
            nodes[i]->Reset();
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
        funcNodes[nNodes - 1]->UpdateGradient(1);

        std::size_t i = nNodes - 1;
        while (true)
        {
            ComputationGraphNode *input1 = funcNodes[i]->GetFirstInput();
            ComputationGraphNode *input2 = funcNodes[i]->GetSecondInput();
            auto gradients = funcNodes[i]->Backward();
            input1->UpdateGradient(gradients[0] * funcNodes[i]->GetPartialGradient());
            input2->UpdateGradient(gradients[1] * funcNodes[i]->GetPartialGradient());
            if (i == 0)
                break;
            i--;
        }
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < nodes.Size(); i++)
            nodes[i]->MarkGradientValuated();
    }

    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraph<T>::CreateConstantNode(const T &value, const std::string &name)
    {
        ConstantNode *newNode = new ConstantNode(value, name);
        auto nodeIndex = nodes.Size();
        nodes.Append(newNode);
        return ComputationGraphNodeHandler<T>(this, nodeIndex);
    }

    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraph<T>::CreateVariableNode(const T &value, const std::string &name)
    {
        VariableNode *newNode = new VariableNode(value, name);
        auto nodeIndex = nodes.Size();
        nodes.Append(newNode);
        return ComputationGraphNodeHandler<T>(this, nodeIndex, true);
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
    T ComputationGraph<T>::GetValue(const ComputationGraphNodeHandler<T> &handler) const
    {
        if (handler.index > this->nodes.Size() - 1)
            throw Exceptions::NodeNotFound(
                "ComputationGraph: Node could not be found.");
        return nodes[handler.index]->Forward();
    }

    template <class T>
    void ComputationGraph<T>::SetValue(const ComputationGraphNodeHandler<T> &handler, const T &newValue) const
    {
        if (handler.index > this->nodes.Size() - 1)
            throw Exceptions::NodeNotFound(
                "ComputationGraph: Node could not be found.");
        try
        {
            auto variable = dynamic_cast<VariableNode *>(nodes[handler.index]);
            variable->SetValue(newValue);
            reset();
        }
        catch(const std::bad_cast&)
        {
            throw Exceptions::InvalidArgument(
                "ComputationGraph: The handler does not represent a variable node.");
        }
    }

    template <class T>
    T ComputationGraph<T>::GetGradient(const ComputationGraphNodeHandler<T> &handler) const
    {
        if (handler.index > this->nodes.Size() - 1)
            throw Exceptions::NodeNotFound(
                "ComputationGraph: Node could not be found.");
        try
        {
            return nodes[handler.index]->GetGradient();
        }
        catch (const Exceptions::GradientNotEvaluated &e)
        {
            throw e;
        }
    }

    template <class T>
    std::string ComputationGraph<T>::GetNodeName(const ComputationGraphNodeHandler<T> &handler) const
    {
        if (handler.index > this->nodes.Size() - 1)
            throw Exceptions::NodeNotFound(
                "ComputationGraph: Node could not be found.");
        return nodes[handler.index]->GetName();
    }

    template <class T>
    std::string ComputationGraph<T>::ToString() const
    {
        std::stringstream ss;
        ss << "ComputationGraph {\n";
        for (std::size_t i = 0; i < nodes.Size(); i++)
        {
            ss << "  ";
            auto nodeString = nodes[i]->ToString();
            for (std::size_t j = 0; j < nodeString.length(); j++)
            {
                ss << nodeString[j];
                if (nodeString[j] == '\n')
                    ss << "  ";
            }
            if (i < nodes.Size() - 1)
                ss << ",\n";
        }
        ss << "\n}";
        return ss.str();
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &stream, const ComputationGraph<T> &graph)
    {
        stream << graph.ToString();
        return stream;
    }

    template <class T>
    ComputationGraph<T>::ComputationGraphNode::ComputationGraphNode(const std::string &nodeName)
        : name(nodeName), gradientValuated(false), gradient(0) {}

    template <class T>
    std::string ComputationGraph<T>::ComputationGraphNode::GetName() const { return name; }

    template <class T>
    void ComputationGraph<T>::ComputationGraphNode::Reset()
    {
        gradientValuated = false;
        gradient = 0;
    }

    template <class T>
    void ComputationGraph<T>::ComputationGraphNode::UpdateGradient(T partialGradient)
    {
        gradient += partialGradient;
    }

    template <class T>
    void ComputationGraph<T>::ComputationGraphNode::MarkGradientValuated()
    {
        gradientValuated = true;
    }

    template <class T>
    T ComputationGraph<T>::ComputationGraphNode::GetGradient() const
    {
        if (gradientValuated)
            return gradient;
        throw Exceptions::GradientNotEvaluated(
            "ComputationGraphNode: Gradient has not been computed.");
    }

    template <class T>
    T ComputationGraph<T>::ComputationGraphNode::GetPartialGradient() const
    {
        return gradient;
    }

    template <class T>
    std::string ComputationGraph<T>::ComputationGraphNode::ToString() const
    {
        std::stringstream ss;
        ss << name << " {\n";
        ss << "  gradient: ";
        this->gradientValuated ? ss << this->gradient : ss << "NOT VALUATED";
        ss << "\n}";
        return ss.str();
    }

    template <class T>
    ComputationGraph<T>::ConstantNode::ConstantNode(const T &value, const std::string &nodeName) : ComputationGraphNode(nodeName)
    {
        this->value = value;
    }

    template <class T>
    T ComputationGraph<T>::ConstantNode::Forward() { return value; }

    template <class T>
    Tuple<T> ComputationGraph<T>::ConstantNode::Backward() { return Tuple<T>({1}); }

    template <class T>
    std::string ComputationGraph<T>::ConstantNode::ToString() const
    {
        std::stringstream ss;
        ss << this->name << " {\n";
        ss << "  value: " << value << ",\n";
        ss << "  gradient: ";
        this->gradientValuated ? ss << this->gradient : ss << "NOT VALUATED";
        ss << "\n}";
        return ss.str();
    }

    template <class T>
    ComputationGraph<T>::VariableNode::VariableNode(const T &value, const std::string &nodeName) : ConstantNode(value, nodeName) {}

    template <class T>
    void ComputationGraph<T>::VariableNode::SetValue(const T &newValue)
    {
        this->value = newValue;
    }

    template <class T>
    ComputationGraph<T>::FunctionNode::FunctionNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName)
        : ComputationGraphNode(nodeName), firstInput(input1), secondInput(input2) {}

    template <class T>
    typename ComputationGraph<T>::ComputationGraphNode *ComputationGraph<T>::FunctionNode::GetFirstInput() const { return firstInput; }

    template <class T>
    typename ComputationGraph<T>::ComputationGraphNode *ComputationGraph<T>::FunctionNode::GetSecondInput() const { return secondInput; }

    template <class T>
    ComputationGraph<T>::AddNode::AddNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::AddNode::Forward()
    {
        return this->firstInput->Forward() + this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ComputationGraph<T>::AddNode::Backward() { return Tuple<T>({1, 1}); }

    template <class T>
    ComputationGraph<T>::MinusNode::MinusNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::MinusNode::Forward()
    {
        return this->firstInput->Forward() - this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ComputationGraph<T>::MinusNode::Backward() { return Tuple<T>({1, -1}); }

    template <class T>
    ComputationGraph<T>::MultiplyNode::MultiplyNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::MultiplyNode::Forward()
    {
        return this->firstInput->Forward() * this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ComputationGraph<T>::MultiplyNode::Backward()
    {
        return Tuple<T>({this->secondInput->Forward(), this->firstInput->Forward()});
    }

    template <class T>
    ComputationGraph<T>::DivideNode::DivideNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::DivideNode::Forward()
    {
        return this->firstInput->Forward() / this->secondInput->Forward();
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
    ComputationGraph<T>::PowerNode::PowerNode(ComputationGraphNode *input1, ComputationGraphNode *input2, const std::string &nodeName)
        : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ComputationGraph<T>::PowerNode::Forward()
    {
        return Math::Power(this->firstInput->Forward(), this->secondInput->Forward());
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
    ComputationGraphNodeHandler<T>::ComputationGraphNodeHandler(ComputationGraph<T> *ownerGraph, std::size_t nodeIndex, bool isVariable)
        : index(nodeIndex), graph(ownerGraph), isVariable(isVariable) {}

    template <class T>
    T ComputationGraphNodeHandler<T>::Forward() const
    {
        return graph->GetValue(*this);
    }

    template <class T>
    T ComputationGraphNodeHandler<T>::Gradient() const
    {
        return graph->GetGradient(*this);
    }

    template <class T>
    std::string ComputationGraphNodeHandler<T>::GetNodeName() const
    {
        return graph->GetNodeName(*this);
    }

    template <class T>
    bool ComputationGraphNodeHandler<T>::IsVariable() const { return isVariable; }

    template <class T>
    void ComputationGraphNodeHandler<T>::SetValue(const T &newValue) const
    {
        graph->SetValue(*this, newValue);
    }

    template <class T>
    ComputationGraphNodeHandler<T> ComputationGraphNodeHandler<T>::operator+(const ComputationGraphNodeHandler<T> &other) const
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
    ComputationGraphNodeHandler<T> ComputationGraphNodeHandler<T>::operator-(const ComputationGraphNodeHandler<T> &other) const
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
    ComputationGraphNodeHandler<T> ComputationGraphNodeHandler<T>::operator*(const ComputationGraphNodeHandler<T> &other) const
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
    ComputationGraphNodeHandler<T> ComputationGraphNodeHandler<T>::operator/(const ComputationGraphNodeHandler<T> &other) const
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