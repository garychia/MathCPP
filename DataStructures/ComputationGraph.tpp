namespace DataStructure
{
    template <class T>
    ComputationGraph<T>::ComputationGraph() : nodes() {}

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
    T ComputationGraph<T>::GetValue(const ComputationGraphNodeHandler<T> &handler) const
    {
        if (handler.index > this->nodes.Size() - 1)
            throw Exceptions::InvalidArgument(
                "ComputationGraph: Node could not be found.");
        return nodes[handler.index]->Forward();
    }

    template <class T>
    T ComputationGraph<T>::GetGradient(const ComputationGraphNodeHandler<T> &handler) const
    {
        if (handler.index > this->nodes.Size() - 1)
            throw Exceptions::InvalidArgument(
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
            throw Exceptions::InvalidArgument(
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
    bool ComputationGraph<T>::ComputationGraphNode::IsGradientValuated() const
    {
        return gradientValuated;
    }

    template <class T>
    void ComputationGraph<T>::ComputationGraphNode::MarkGradientValuated()
    {
        gradientValuated = true;
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