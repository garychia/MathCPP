#ifndef ML_ALGS_HPP
#define ML_ALGS_HPP

#include <functional>

#include "Vector.hpp"
#include "Matrix.hpp"
#include "List.hpp"

using namespace DataStructures;

namespace MachineLearning
{
    /*
    Sign function.
    @param value a value.
    @return +1 if the value is positive, or 0 if the value is 0. -1, otherwise.
    */
    template <class T>
    T Sign(T value);

    /*
    Generates a Vector that represents a given value using One Hot encoding.
    @param value an unsigned integer to be encoded.
    @param k the maximum value the value can take.
    @return a Vector of K binary features that represents the given value.
    */
    Vector<int> OneHot(std::size_t value, std::size_t k);

    /**
     * Computes the Hinge Loss
     * @param prediction the prediction.
     * @param label the expected outcome.
     * @return the hinge loss given the prediction and label. 
     **/
    template <class T>
    T HingeLoss(const T &prediction, const T &label);

    /**
     * Computes the gradient of Hinge Loss with respect to model weights.
     * @param input the input to the model.
     * @param prediction the prediction the model has made.
     * @param label the expected output.
     * @return the gradient of Hinge Loss with resect to model weights.
     **/
    template <class InputType, class LabelType>
    InputType HingeLossGradient(const InputType &input, const LabelType &prediction, const LabelType &label);

    /*
    The Perceptron Algorithm.
    @param data a Matrix with n rows and d columns where n is the number of data points
    and d is the dimensions of them.
    @param labels a Vector with a single row and n columns indicating the labels of the
    data points.
    @param T the number of iterations to repeat the algorithm (default: 1000).
    @return a Vector packed with the parameters (theta and offset at the end).
    */
    template <class DataType, class LabelType>
    Vector<double> Perceptron(const Matrix<DataType> &data, const Vector<LabelType> &labels, std::size_t T = 1000);

    /*
    The Averaged Perceptron Algorithm.
    @param data a Matrix with n rows and d columns where n is the number of data points
    and d is the dimensions of them.
    @param labels a Vector with a single row and n columns indicating the labels of the
    data points.
    @param T the number of iterations to repeat the algorithm (default: 1000).
    @return a Vector packed with the averaged parameters (theta and offset at the end).
    */
    template <class DataType, class LabelType>
    Vector<double> AveragedPerceptron(const Matrix<DataType> &data, const Vector<LabelType> &labels, std::size_t T = 1000);

    /*
    Gradient Descent
    @param f a function whose output will be minimized.
    @param df a function that outputs the gradient of f.
    @param initialX the initial input to f.
    @param stepFunc a function that takes the current number of iterations and
    returns the step size the gradient descent algorithm should take.
    @param iterations the number of iterations to perform gradient descent.
    @param recordHistory a bool indicates whether the input to and the output
    of f are recorded in xHistory and outputHistory, respectively.
    @param xHistory a pointer to the List used to record the inputs to f
    during the process if recordHistory is set to true.
    @param outputHistory a pointer to the List used to record the outputs of
    f during the process if recordHistory is set to true.
    @return the value of the input of f at the final step of gradient descent.
    */
    template <class InputType, class OutputType, class StepType>
    InputType GradientDescent(
        const std::function<OutputType(const InputType &)> &f,
        const std::function<InputType(const InputType &)> &df,
        const InputType &initialX,
        const std::function<StepType(std::size_t)> &stepFunc,
        std::size_t iterations,
        bool recordHistory = false,
        List<InputType> *xHistory = nullptr,
        List<OutputType> *outputHistory = nullptr);

} // namespace MachineLearning

#include "Algorthms.tpp"

#endif