#ifndef LAYER_H
#define LAYER_H

#include <string>
#include "../Activation/activation.hpp"
#include "../Matrix/matrix.hpp"

typedef float (*FloatToFloatFunc)(float);

class Layer {
public:
    Layer(unsigned size) : size(size), neurons(Matrix<float>(size, 1)) {}
    virtual void summary() = 0;
    virtual void initWeights(unsigned prevSize) = 0;
    Matrix<float>& getNeurons() { return this->neurons; }
    void setNeurons(const Matrix<float>& m) { this->neurons = m; }
    virtual Matrix<float>& getWeigths() { return this->neurons; }
    virtual void setWeights(const Matrix<float>& m) { (void)m; }
    virtual Matrix<float>& getBias() { return this->neurons; }
    virtual void setBias(const Matrix<float>& m) { (void)m; }
    virtual FloatToFloatFunc getActivation() { std::cout << "getActivation() is virtual." << std::endl; throw; }
    virtual FloatToFloatFunc getDerivation() { std::cout << "getDerivation() is virtual." << std::endl; throw; }
    unsigned getSize() { return this->size; }

protected:
    unsigned size;
    Matrix<float> neurons;
};

#endif
