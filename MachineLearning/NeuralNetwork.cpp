#include "NeuralNetwork.hpp"
#include "Exceptions.hpp"

#include "LinearLayer.hpp"
#include "ReLULayer.hpp"
#include "SoftMaxLayer.hpp"
#include "TanhLayer.hpp"

#include <ostream>

namespace MachineLearning
{
    NeuralNetwork::NeuralNetwork() : layers()
    {
    }

    NeuralNetwork::NeuralNetwork(const List<LayerType> &layerTypes, const List<Tuple<unsigned int>> &shapes)
    {
        if (layerTypes.Size() != shapes.Size())
            throw Exceptions::InvalidArgument(
                "NeuralNetwork: Input and output size must be specified for each layer.");
        for (std::size_t i = 0; i < layerTypes.Size(); i++)
        {
            if (shapes[i].Size() != 2)
                throw Exceptions::InvalidArgument(
                    "NeuralNetwork: Input and output size for each layer must be specified exactly by 2 numbers.");
            const auto inputSize = shapes[i][0];
            const auto outputSize = shapes[i][1];
            switch (layerTypes[i])
            {
            case LayerType::Linear:
            {
                layers.Append(new LinearLayer(inputSize, outputSize));
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
    }

    NeuralNetwork::~NeuralNetwork()
    {
        for (std::size_t i = 0; i < layers.Size(); i++)
            delete layers[i];
        layers.Clear();
    }

    Matrix<double> NeuralNetwork::Predict(const Matrix<double> &input)
    {
        Matrix<double> output = input;
        for (std::size_t i = 0; i < layers.Size(); i++)
            output = layers[i]->Forward(output);
        return output;
    }

    std::string NeuralNetwork::ToString() const
    {
        std::stringstream ss;
        const auto nLayers = layers.Size();
        ss << "NeuralNetwork:\n";
        ss << "Number of Layers: " << nLayers;
        if (nLayers > 0)
        {
            ss << std::endl << "Layers:\n";
            for (std::size_t i = 0; i < nLayers; i++)
            {
                ss << layers[i]->ToString() << std::endl;
                ss << "------------------------------------------";
                if (i < nLayers - 1)
                    ss << std::endl;
            }
        }
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
} // namespace MachineLearning
