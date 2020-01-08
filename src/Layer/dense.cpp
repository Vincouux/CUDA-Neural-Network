#include "dense.hpp"

Dense::Dense(unsigned size, ActivationFunction activation) : Layer(size), weights(Matrix<float>(0, 0)), bias(Matrix<float>(0, 0)), activation(Activation(activation)) {}

void Dense::initWeights(unsigned prevSize) {
    this->weights = Matrix<float>(this->size, prevSize);
}

void Dense::summary() {
    std::cout << "Dense Layer " << this->size << std::endl;
}
