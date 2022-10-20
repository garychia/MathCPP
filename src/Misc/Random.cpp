#include "Random.hpp"
#include "Math.hpp"

#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int Random::seed = 0;

bool Random::useSeed = false;

void Random::SetSeed(int randomSeed) {
  seed = randomSeed;
  useSeed = true;
}

void Random::UnsetSeed() { useSeed = false; }

int Random::Generate() {
  int randomValue;
#pragma omp critical
  {
#ifdef _OPENMP
    srand((useSeed ? seed : clock()) + omp_get_thread_num());
#else
    srand(useSeed ? seed : clock());
#endif
    randomValue = rand();
  }
  return randomValue;
}

int Random::IntRange(int low, int high) {
  return Generate() % (high - low) + low;
}

unsigned int Random::Choose(unsigned int nElements,
                            const DataStructures::List<double> &prob) {
  if (prob.IsEmpty())
    return IntRange(0, nElements);
  const double randomProb = (double)(Generate() % 1001) / 1000;
  double currentProbability = 0.0;
  for (std::size_t i = 0; i < prob.Size(); i++) {
    currentProbability += prob[i];
    if (randomProb <= currentProbability)
      return i;
  }
  return prob.Size() - 1;
}

double Random::NormalDistribution(double mean, double standard,
                                  unsigned int nSamples) {
  const auto rangeMin = mean - 3 * standard;
  const auto rangeMax = mean + 3 * standard;
  const auto rangeSize = rangeMax - rangeMin;
  const auto stepSize = rangeSize / nSamples;
  DataStructures::List<double> distribution;
  DataStructures::List<double> values;
  for (auto i = rangeMin; i <= rangeMax; i += stepSize) {
    distribution.Append(Math::Gauss(i, mean, standard) * stepSize);
    values.Append(i);
  }
  return values[Choose(distribution.Size(), distribution)];
}
