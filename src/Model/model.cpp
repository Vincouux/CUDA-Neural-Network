#include "model.hpp"

Model() {
    this->layers = std::vectors<Layer>();
}

void Model::add(Layer layer) {
    this->layers.push_back(layer);
}

void Model::compile() {
    this->depth = this->layers.size();
    for (unsigned i = 1; i < this->depth; i++) {
        this->layers[i]->initWeights(this->layers[i - 1]);
    }
}
