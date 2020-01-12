#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <iostream>
#include "../Layer/layer.hpp"

class Model {
public:
    Model();
    ~Model();
    void add(Layer* layer);
    void fit(const Matrix<float>& X, const Matrix<float>& Y);
    void forward();
    void backward(const Matrix<float>& Y);
    Matrix<float> predict(const Matrix<float>& X);
    void compile();
    void summary();


    std::vector<Layer*> layers;
    unsigned depth;
};

#endif
