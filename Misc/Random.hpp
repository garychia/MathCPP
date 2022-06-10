#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "List.hpp"

class Random
{
private:
    static int seed;

    static void UpdateSeed();
public:
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
     * @return a random number sampled from a normal distribution.
     **/
    static double NormalDistribution(double mean, double standard);
};

#endif // RANDOM_HPP