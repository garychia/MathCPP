#include "Random.hpp"
#include "Math.hpp"

#include <time.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int Random::seed = 0;

void Random::UpdateSeed()
{
    seed = (seed << 3) % 17 + 19;
}

int Random::IntRange(int low, int high)
{
    int randomValue;
#pragma omp critical
{
    srand(time(NULL) + seed);
    UpdateSeed();
    randomValue = rand();
}
    return randomValue % (high - low) + low;
}

unsigned int Random::Choose(unsigned int nElements, const DataStructures::List<double> &prob)
{
    if (prob.IsEmpty())
        return IntRange(0, nElements);
    int randomValue;
#pragma omp critical
{
    randomValue = rand();
}
    const double randomProb = (double)(randomValue % 1000001) / 1000000.0;
    double currentProbability = 0.0;
    for (std::size_t i = 0; i < prob.Size(); i++)
    {
        currentProbability += prob[i];
        if (randomProb <= currentProbability)
            return i;
    }
    return prob.Size() - 1;
}

double Random::NormalDistribution(double mean, double standard)
{
    const auto rangeMin = mean - 3 * standard;
    const auto rangeMax = mean + 3 * standard;
    const double stepSize = 0.01;
    DataStructures::List<double> distribution;
    for (auto i = rangeMin; i <= rangeMax; i += stepSize)
        distribution.Append(Math::Gauss(i, mean, standard));
    return distribution[Choose(distribution.Size(), distribution)];
}