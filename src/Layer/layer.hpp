#ifndef LAYER_H
#define LAYER_H

#include <string>
#include "../Activation/activation.hpp"
#include "../Matrix/matrix.hpp"

class Layer {
public:
    Layer(unsigned size) : size(size), neurons(Matrix<float>(size, 1)) {}
    virtual void summary() = 0;
    virtual void initWeights(unsigned prevSize) = 0;
    Matrix<float> getNeurons() { return this->neurons; }
    void setNeurons(Matrix<float> m) { this->neurons = m; }
    unsigned getSize() { return this->size; }

protected:
    unsigned size;
    Matrix<float> neurons;
};

#endif
