#ifndef LAYER_H
#define LAYER_H

#include <string>
#include "../Activation/activation.hpp"
#include "../Matrix/matrix.hpp"

enum Type {
    Input,
    Dense
};

class Layer {
public:
    Layer(unsigned size, Type type, ActivationFunction activation);
    ~Layer();
    void initWeights(Layer prevLayer);
    std::string getType();
    void summary();

private:
    unsigned size;
    Type type;
    Activation activation;
    Matrix<float>* weights;
    Matrix<float>* neurons;
};

#endif
