#include "dense.hpp"

Dense::Dense(unsigned size, ActivationFunction activation) : Layer(size), weights(Matrix<float>()), bias(Matrix<float>(size, 1)), activation(Activation(activation)) {}

void Dense::initWeights(unsigned prevSize) {
    this->weights = Matrix<float>(this->size, prevSize, -0.5f, 0.5f);
}

Matrix<float>& Dense::getWeigths() {
    return this->weights;
}

void Dense::setWeights(const Matrix<float>& m) {
    this->weights = m;
}

Matrix<float>& Dense::getBias() {
    return this->bias;
}
void Dense::setBias(const Matrix<float>& m) {
    this->bias = m;
}

FloatToFloatFunc Dense::getActivation() {
    return this->activation.activate;
}

FloatToFloatFunc Dense::getDerivation() {
    return this->activation.derivate;
}

void Dense::summary() {
    std::cout << "Dense Layer " << this->size << std::endl << std::endl;
    std::cout << "Neurons: " << std::endl;
    this->neurons.display();
    std::cout << std::endl << "Weights: " << std::endl;
    this->weights.display();
    std::cout << std::endl << "Bias: " << std::endl;
    this->bias.display();
    std::cout << "---------------------" << std::endl << std::endl;
}
