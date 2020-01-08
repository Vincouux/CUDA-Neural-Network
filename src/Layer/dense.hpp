#ifndef DENSE_H
#define DENSE_H

#include <string>
#include "layer.hpp"

class Dense: public Layer {
public:
    Dense(unsigned size, ActivationFunction activation);
    void initWeights(unsigned prevSize);
    void summary();

private:
    Matrix<float> weights;
    Matrix<float> bias;
    Activation activation;
};

#endif
