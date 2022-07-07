#include "NeuralNetwork.hpp"
#include "Exceptions.hpp"

#include "Layers/LinearLayer.hpp"
#include "Layers/ReLULayer.hpp"
#include "Layers/SoftMaxLayer.hpp"
#include "Layers/TanhLayer.hpp"
#include "Layers/NLLLayer.hpp"

#include "Random.hpp"

#include <ostream>

namespace MachineLearning
{
    NeuralNetwork::NeuralNetwork() : layers()
    {
    }

    NeuralNetwork::NeuralNetwork(const List<LayerType> &layerTypes, const List<Tuple<unsigned int>> &shapes, LossType lossType) : layers()
    {
        if (layerTypes.Size() != shapes.Size())
            throw Exceptions::InvalidArgument(
                "NeuralNetwork: Input and output size must be specified for each layer.");
        for (std::size_t i = 0; i < layerTypes.Size(); i++)
        {
            switch (layerTypes[i])
            {
            case LayerType::Linear:
            {
                if (shapes[i].Size() != 2)
                    throw Exceptions::InvalidArgument(
                        "NeuralNetwork: the input and output sizes of a LinearLayer must be specified.");
                layers.Append(new LinearLayer(shapes[i][0], shapes[i][1]));
                break;
            }
            case LayerType::ReLU:
            {
                layers.Append(new ReLULayer());
                break;
            }
            case LayerType::SoftMax:
            {
                layers.Append(new SoftMaxLayer());
                break;
            }
            case LayerType::Tanh:
            {
                layers.Append(new TanhLayer());
                break;
            }
            default:
            {
                std::stringstream ss;
                ss << "NeuralNetwork: Unexpected LayerType: " << layerTypes[i] << ".";
                throw Exceptions::InvalidArgument(ss.str());
            }
            }
        }

        switch (lossType)
        {
        case LossType::NLL:
            lossLayer = new NLLLayer();
            break;
        default:
        {
            std::stringstream ss;
            ss << "NeuralNetwork: Unexpected LossType: " << lossType << ".";
            throw Exceptions::InvalidArgument(ss.str());
        }
        }
    }

    NeuralNetwork::~NeuralNetwork()
    {
        for (std::size_t i = 0; i < layers.Size(); i++)
            delete layers[i];
        layers.Clear();
        delete lossLayer;
    }

    Matrix<double> NeuralNetwork::Predict(const Matrix<double> &input)
    {
        Matrix<double> output = input;
        for (std::size_t i = 0; i < layers.Size(); i++)
            output = layers[i]->Forward(output);
        return output;
    }

    void NeuralNetwork::Learn(const Matrix<double> &derivative, const double &learningRate)
    {
        if (!layers.IsEmpty())
        {
            Matrix<double> currentDerivative = derivative;
            std::size_t i = layers.Size() - 1;
            while (true)
            {
                currentDerivative = layers[i]->Backward(currentDerivative);
                layers[i]->UpdateWeights(learningRate);
                if (i == 0)
                    break;
                i--;
            }
        }
    }

    List<double> NeuralNetwork::Train(const Matrix<double> &trainingData, const Matrix<double> &labels, unsigned int epochs, double learningRate)
    {
        List<double> lossHistory;
        for (std::size_t i = 0; i < epochs; i++)
        {
            const auto pred = Predict(trainingData);
            lossHistory.Append(lossLayer->ComputeLoss(pred, labels));
            const auto derivative = lossLayer->Backward(pred, labels);
            Learn(derivative, learningRate);
        }
        return lossHistory;
    }

    std::string NeuralNetwork::ToString() const
    {
        std::stringstream ss;
        const auto nLayers = layers.Size();
        ss << "NeuralNetwork: {\n";
        ss << "  Number of Layers: " << nLayers << ",";
        if (nLayers > 0)
        {
            ss << std::endl
               << "  Layers: {\n";
            for (std::size_t i = 0; i < nLayers; i++)
            {
                ss << "    Layer " << i + 1 << ": {\n";
                ss << "      ";
                for (const auto &c : layers[i]->ToString())
                {
                    ss << c;
                    if (c == '\n')
                        ss << "      ";
                }
                ss << "\n    }";
                if (i < nLayers - 1)
                    ss << ",";
                ss << "\n";
            }
            ss << "  },\n";
        }
        ss << "  Loss Layer: " << lossLayer->ToString() << std::endl;
        ss << "}";
        return ss.str();
    }

    std::ostream &operator<<(std::ostream &stream, const NeuralNetwork &network)
    {
        stream << network.ToString();
        return stream;
    }

    std::ostream &operator<<(std::ostream &stream, const NeuralNetwork::LayerType &layerType)
    {
        stream << static_cast<int>(layerType);
        return stream;
    }

    std::ostream &operator<<(std::ostream &stream, const NeuralNetwork::LossType &lossType)
    {
        stream << static_cast<int>(lossType);
        return stream;
    }
} // namespace MachineLearning
