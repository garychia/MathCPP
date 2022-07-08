#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "List.hpp"

class Random
{
private:
    static int seed;

    static bool useSeed;

    static int Generate();
public:
    /**
     * Set up the random seed.
     * @param randomSeed the seed.
     **/
    static void SetSeed(int randomSeed);

    /**
     * Remove the seed.
     **/
    static void UnsetSeed();

    /**
     * Generate a random integer.
     * @param low the minimum value of the output (inclusive).
     * @param high the maximum value of the output (exclusive).
     * @return a random integer.
     **/
    static int IntRange(int low, int high);

    /**
     * Choose an integer based on a probability distribution.
     * @param nElements the total number of elements.
     * @param prob the probability distribution.
     * @return the index of the element selected. (0 for the first one). 
     **/
    static unsigned int Choose(unsigned int nElements, const DataStructures::List<double> &prob = DataStructures::List<double>());

    /**
     * Generate a random number sampled from a normal distribution.
     * @param mean the mean of the normal distribution.
     * @param standard the standard deviation of the distribution.
     * @param nSamples the total number of samples from the distribution to choose from.
     * @return a random number sampled from a normal distribution.
     **/
    static double NormalDistribution(double mean, double standard, unsigned int nSamples = 1000);
};

#endif // RANDOM_HPP