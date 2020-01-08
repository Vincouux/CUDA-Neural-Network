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
    void fit(Matrix<float> X, Matrix<float> Y);
    void compile();
    void summary();

private:
    std::vector<Layer*> layers;
    unsigned depth;
};

#endif
