#ifndef ACTIVATION_H
#define ACTIVATION_H

enum ActivationFunction {
    Sigmoid,
    Relu,
    Linear,
    TanH
};

class Activation {
public:
    Activation(enum ActivationFunction);
    ~Activation();
    Matrix* activate(Matrix* m);
    Matrix* derivate(Matrix* m);

private:
    ActivationFunction activationFunction;
};

#endif
