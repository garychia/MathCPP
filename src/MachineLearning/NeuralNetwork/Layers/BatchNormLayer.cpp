#include "BatchNormLayer.hpp"

#include "Math.hpp"

namespace MachineLearning {
BatchNormLayer::BatchNormLayer(std::size_t inputSize)
    : NeuralLayer(), mean(), variance(), normalized(), dScale(), dShift() {
  const auto sigma = Math::Power<double, double>(inputSize, -0.5);
  scale = Matrix<double>(inputSize, 1).Map([&sigma](const double &) {
    return Random::NormalDistribution(0, sigma);
  });
  shift = Matrix<double>(inputSize, 1);
}

Matrix<double> BatchNormLayer::Forward(const Matrix<double> &input) {
  this->input = input;
  const auto batchSize = input.Shape()[1];
  mean = input.Sum(false) / batchSize;
  variance = (input - mean)
                 .Map([](const double &e) { return e * e; })
                 .Sum(false)
                 .Divide(batchSize);
  normalized = (input - mean) / variance.Map([](const double &e) {
    return Math::Power(e, 0.5) + 1E-10;
  });
  this->output = normalized.Scale(scale) + shift;
  return this->output;
}

Matrix<double> BatchNormLayer::Backward(const Matrix<double> &derivative) {
  this->dScale = derivative.Scale(normalized).Sum(false);
  this->dShift = derivative.Sum(false);
  const auto batchSize = this->input.Shape()[1];
  const auto inversedSTD = variance.Map(
      [](const double &e) { return Math::Power(e, -0.5) + 1E-10; });
  const auto inputMeanDiff = this->input - mean;
  const auto dNorm = derivative.Scale(this->scale);
  const auto dVariance = dNorm.Scale(inputMeanDiff)
                             .Scale(-0.5 * inversedSTD.Map([](const double &e) {
                               return e * e * e;
                             }))

                             .Sum(false);
  const auto dNormInversedSTDProduct = dNorm.Scale(inversedSTD);
  const auto dMean =
      (dNormInversedSTDProduct * -1).Sum(false) +
      (dVariance * (-2 / batchSize)).Scale(inputMeanDiff.Sum(false));
  const auto dInput = inputMeanDiff.Scale(dVariance * 2 / batchSize) +
                      dNormInversedSTDProduct + mean / batchSize;
  return dInput;
}

void BatchNormLayer::UpdateWeights(const double &learningRate) {
  scale -= dScale * learningRate;
  shift -= dShift * learningRate;
}

std::string BatchNormLayer::ToString() const {
  std::stringstream ss;
  ss << "BatchNormLayer: {" << std::endl;
  ss << "  Input: {\n";
  ss << "    ";
  for (const auto &c : this->input.ToString()) {
    ss << c;
    if (c == '\n')
      ss << "    ";
  }
  ss << "\n  },\n";
  ss << "  Output: {\n";
  ss << "    ";
  for (const auto &c : this->output.ToString()) {
    ss << c;
    if (c == '\n')
      ss << "    ";
  }
  ss << "\n  },\n";
  ss << "  Scaling Factor: {\n";
  ss << "    ";
  for (const auto &c : scale.ToString()) {
    ss << c;
    if (c == '\n')
      ss << "    ";
  }
  ss << "\n  },\n";
  ss << "  Shifting Factor: {\n";
  ss << "    ";
  for (const auto &c : shift.ToString()) {
    ss << c;
    if (c == '\n')
      ss << "    ";
  }
  ss << "\n  }\n";
  ss << "}";
  return ss.str();
}
} // namespace MachineLearning