#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../Matrix/matrix.hpp"

enum ActivationFunction {
    Sigmoid,
    Relu,
    Linear,
    TanH,
    LeakyRelu
};

class Activation {
public:
    Activation(ActivationFunction activation);
    ~Activation();
    ActivationFunction activationFunction;
    float (*activate)(float);
    float (*derivate)(float);
    void summary();

private:
    std::string name;
};

#endif
