namespace DataStructure
{
    template <class T>
    T ScalerComputationGraph<T>::Forward()
    {
        if (this->funcNodes.IsEmpty())
            return 0;
        return this->funcNodes[this->funcNodes.Size() - 1]->Forward();
    }

    template <class T>
    void ScalerComputationGraph<T>::Backward()
    {
        if (this->funcNodes.IsEmpty())
            return;
        const std::size_t nNodes = this->funcNodes.Size();
        this->funcNodes[nNodes - 1]->UpdateGradient(1);

        std::size_t i = nNodes - 1;
        while (true)
        {
            ComputationGraphNode* input1 = this->funcNodes[i]->GetFirstInput();
            ComputationGraphNode* input2 = this->funcNodes[i]->GetSecondInput();
            auto gradients = this->funcNodes[i]->Backward();
            input1->UpdateGradient(gradients[0] * this->funcNodes[i]->GetPartialGradient());
            input2->UpdateGradient(gradients[1] * this->funcNodes[i]->GetPartialGradient());
            if (i == 0)
                break;
            i--;
        }
#pragma omp parallel for schedule(dynamic)
        for (std::size_t i = 0; i < this->nodes.Size(); i++)
            this->nodes[i]->MarkGradientValuated();
    }

    template <class T>
    void ScalerComputationGraph<T>::SetValue(const ComputationGraphNodeHandler<T>& handler, const T& newValue) const
    {
        if (handler.index > this->nodes.Size() - 1)
            throw Exceptions::InvalidArgument(
                "ComputationGraph: Node could not be found.");
        try
        {
            auto variable = dynamic_cast<VariableNode*>(nodes[handler.index]);
            variable->SetValue(newValue);
            reset();
        }
        catch (const std::bad_cast&)
        {
            throw Exceptions::InvalidArgument(
                "ComputationGraph: The handler does not represent a variable node.");
        }
    }

    template <class T>
    ComputationGraphNodeHandler<T> ScalerComputationGraph<T>::CreateConstantNode(const T& value, const std::string& name)
    {
        ConstantNode* newNode = new ConstantNode(value, name);
        auto nodeIndex = nodes.Size();
        nodes.Append(newNode);
        return ComputationGraphNodeHandler<T>(this, nodeIndex);
    }

    template <class T>
    ComputationGraphNodeHandler<T> ScalerComputationGraph<T>::CreateVariableNode(const T& value, const std::string& name)
    {
        VariableNode* newNode = new VariableNode(value, name);
        auto nodeIndex = nodes.Size();
        nodes.Append(newNode);
        return ComputationGraphNodeHandler<T>(this, nodeIndex, true);
    }

    template <class T>
    ComputationGraphNodeHandler<T> ScalerComputationGraph<T>::CreateFunctionNode(
        const ComputationGraphNodeHandler<T>& inputNodeHandler1,
        const ComputationGraphNodeHandler<T>& inputNodeHandler2,
        const GraphOperation& operation,
        const std::string& name)
    {
        FunctionNode* newNode;
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
    ScalerComputationGraph<T>::ScalerComputationGraphNode::ScalerComputationGraphNode(const std::string &nodeName) : ComputationGraphNode(nodeName) { }

    template <class T>
    void ScalerComputationGraph<T>::ScalerComputationGraphNode::Reset()
    {
        this->gradient = 0;
        this->gradientValuated = false;
    }

    template <class T>
    void ScalerComputationGraph<T>::ScalerComputationGraphNode::UpdateGradient(const T& partialGradient)
    {
        gradient += partialGradient;
    }

    template <class T>
    ScalerComputationGraph<T>::ConstantNode::ConstantNode(const T& value, const std::string& nodeName) : ScalerComputationGraphNode(nodeName), value(value) {}

    template <class T>
    T ScalerComputationGraph<T>::ConstantNode::Forward() { return this->value; }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::ConstantNode::Backward() { return Tuple<T>({ 1 }); }

    template <class T>
    ScalerComputationGraph<T>::VariableNode::VariableNode(const T& value, const std::string& nodeName) : ConstantNode(value, nodeName) { }

    template <class T>
    void ScalerComputationGraph<T>::VariableNode::SetValue(const T& newValue)
    {
        this->value = newValue;
    }

    template <class T>
    ScalerComputationGraph<T>::ScalerFunctionNode::ScalerFunctionNode(class ComputationGraphNode* input1, class  ComputationGraphNode* input2, const std::string& nodeName) : FunctionNode(input1, input2, nodeName) {}

    template <class T>
    void ScalerComputationGraph<T>::ScalerFunctionNode::Reset()
    {
        this->gradient = 0;
        this->gradientValuated = false;
    }

    template <class T>
    void ScalerComputationGraph<T>::ScalerFunctionNode::UpdateGradient(const T& partialGradient)
    {
        gradient += partialGradient;
    }

    template <class T>
    ScalerComputationGraph<T>::AddNode::AddNode(class ComputationGraphNode* input1, class ComputationGraphNode* input2, const std::string& nodeName)
        : ScalerFunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::AddNode::Forward()
    {
        return this->firstInput->Forward() + this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::AddNode::Backward() { return Tuple<T>({ 1, 1 }); }

    template <class T>
    ScalerComputationGraph<T>::MinusNode::MinusNode(class ComputationGraphNode* input1, class ComputationGraphNode* input2, const std::string& nodeName)
        : ScalerFunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::MinusNode::Forward()
    {
        return this->firstInput->Forward() - this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::MinusNode::Backward() { return Tuple<T>({ 1, -1 }); }

    template <class T>
    ScalerComputationGraph<T>::MultiplyNode::MultiplyNode(class ComputationGraphNode* input1, class ComputationGraphNode* input2, const std::string& nodeName)
        : ScalerFunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::MultiplyNode::Forward()
    {
        return this->firstInput->Forward() * this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::MultiplyNode::Backward()
    {
        return Tuple<T>({ this->secondInput->Forward(), this->firstInput->Forward() });
    }

    template <class T>
    ScalerComputationGraph<T>::DivideNode::DivideNode(class ComputationGraphNode* input1, class ComputationGraphNode* input2, const std::string& nodeName)
        : ScalerFunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::DivideNode::Forward()
    {
        return this->firstInput->Forward() / this->secondInput->Forward();
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::DivideNode::Backward()
    {
        const T firstInputValue = this->firstInput->Forward();
        const T secondInputValue = this->secondInput->Forward();
        return Tuple<T>({ 1 / secondInputValue,
                         -firstInputValue / (secondInputValue * secondInputValue) });
    }

    template <class T>
    ScalerComputationGraph<T>::PowerNode::PowerNode(class ComputationGraphNode* input1, class ComputationGraphNode* input2, const std::string& nodeName)
        : ScalerFunctionNode(input1, input2, nodeName) {}

    template <class T>
    T ScalerComputationGraph<T>::PowerNode::Forward()
    {
        return Math::Power(this->firstInput->Forward(), this->secondInput->Forward());
    }

    template <class T>
    Tuple<T> ScalerComputationGraph<T>::PowerNode::Backward()
    {
        const T firstInputValue = this->firstInput->Forward();
        const T secondInputValue = this->secondInput->Forward();
        const T powered = Math::Power(firstInputValue, secondInputValue - 1);
        return Tuple<T>({ secondInputValue * powered,
                         powered * firstInputValue * Math::Log(firstInputValue) });
    }
}