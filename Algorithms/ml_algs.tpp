namespace MLAlgs
{
    template <class T>
    T Sign(T value)
    {
        return value == 0 ? 0 : (value > 0 ? 1 : -1);
    }

    Vector<int> OneHot(std::size_t value, std::size_t k)
    {
        Vector<int> encoding(k, 0);
        encoding[value - 1] = 1;
        return encoding;
    }

    template <class DataType, class LabelType>
    Vector<double> Perceptron(const Matrix<DataType> &data, const Vector<LabelType> &labels, std::size_t T)
    {
        const auto dataShape = data.Shape();
        const auto n = dataShape[0];
        const auto d = dataShape[1];
        Vector<double> th(d, 0.0);
        double th0 = 0.0;
        for (std::size_t i = 0; i < T; i++)
            for (std::size_t j = 0; j < n; j++)
                if (Sign(th.Dot(data[j]) + th0) != labels[j])
                {
                    th += data[j] * labels[j];
                    th0 += labels[j];
                }
        return Vector<double>::Combine({std::move(th), Vector<double>(1, th0)});
    }

    template <class DataType, class LabelType>
    Vector<double> AveragedPerceptron(const Matrix<DataType> &data, const Vector<LabelType> &labels, std::size_t T)
    {
        const auto dataShape = data.Shape();
        const auto n = dataShape[0];
        const auto d = dataShape[1];
        Vector<double> th(d, 0.0);
        Vector<double> ths(d, 0.0);
        double th0 = 0.0;
        double th0s = 0.0;
        for (std::size_t i = 0; i < T; i++)
            for (std::size_t j = 0; j < n; j++)
            {
                if (Sign(th.Dot(data[j]) + th0) != labels[j])
                {
                    th += data[j] * labels[j];
                    th0 += labels[j];
                }
                ths += th;
                th0s += th0;
            }
        const auto totalIterations = n * T;
        ths /= totalIterations;
        th0s /= totalIterations;
        return Vector<double>::Combine({std::move(ths), Vector<double>(1, th0s)});
    }

    template <class InputType, class OutputType, class StepType>
    InputType GradientDescent(
        const std::function<OutputType(const InputType &)> &f,
        const std::function<InputType(const InputType &)> &df,
        const InputType &initialX,
        const std::function<StepType(std::size_t)> &stepFunc,
        std::size_t iterations,
        bool recordHistory,
        List<InputType> *xHistory,
        List<OutputType> *outputHistory)
    {
        InputType x = initialX;
        if (recordHistory && xHistory)
            xHistory->Append(x);
        if (recordHistory && outputHistory)
            outputHistory->Append(f(x));
        for (std::size_t i = 1; i < iterations + 1; i++)
        {
            x -= df(x) * stepFunc(i);
            if (recordHistory && xHistory)
                xHistory->Append(x);
            if (recordHistory && outputHistory)
                outputHistory->Append(f(x));
        }
        return x;
    }
}