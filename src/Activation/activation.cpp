#include "activation.hpp"

Activation::Activation() {
    this->activationFunction = Linear;
}
Activation::Activation(ActivationFunction activation) {
    this->activationFunction = activation;
}

Activation::~Activation() {}

Matrix<float> Activation::activate(Matrix<float> m) {
    return m;
}

Matrix<float> Activation::derivate(Matrix<float> m) {
    return m;
}
