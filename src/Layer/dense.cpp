#include "dense.hpp"

Dense::Dense(unsigned size, ActivationFunction activation) : Layer(size), weights(Matrix<float>(0, 0)), bias(Matrix<float>(0, 0)), activation(Activation(activation)) {}

void Dense::initWeights(unsigned prevSize) {
    this->weights = Matrix<float>(this->size, prevSize);
    this->bias = Matrix<float>(this->size, 1);
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

void Dense::summary() {
    std::cout << "Dense Layer " << this->size << std::endl;
    std::cout << "Neurons: " << std::endl;
    this->neurons.display();
    std::cout << "Weights: " << std::endl;
    this->weights.display();
    std::cout << "Bias: " << std::endl;
    this->bias.display();
    std::cout << "---------------------" << std::endl << std::endl;
}
