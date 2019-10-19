#ifndef ACTIVATION_H
#define ACTIVATION_H

class Activation {
public:
  Activation(enum ActivationFunction);
  ~Activation();
  enum ActivationFunction
  {
    Logistic,
    Relu,
    Linear,
    TanH
  };

private:
  ActivationFunction activationFunction;
};

#endif
