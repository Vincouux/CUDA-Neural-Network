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
    Activation(ActivationFunction activation);
    ~Activation();
    ActivationFunction activationFunction;
    float (*activate)(float);
    float (*derivate)(float);
};

#endif
