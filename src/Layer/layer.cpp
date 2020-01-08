#include "layer.hpp"

Layer(size_t size, Type type, Activation activation) {
    this->size = size;
    this->type = type;
    this->activation = activation;
    if (type == Dense) {
        self.neurons = Matrix<float>(this->size, 1);
    } else {
        self.neurons = NULL;
    }
    self.weights = NULL;
}

void initWeights(Layer prevLayer) {
    this->weights = Matrix<float>(2, prevLayer->size);
}
