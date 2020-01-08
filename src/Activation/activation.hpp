#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../Matrix/matrix.hpp"

enum ActivationFunction {
    Sigmoid,
    Relu,
    Linear,
    TanH
};

class Activation {
public:
    Activation();
    Activation(ActivationFunction activation);
    ~Activation();
    Matrix<float> activate(Matrix<float> m);
    Matrix<float> derivate(Matrix<float> m);

private:
    ActivationFunction activationFunction;
};

#endif
