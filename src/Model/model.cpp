#include "model.hpp"

Model::Model() {
    this->layers = std::vector<Layer>();
    this->depth = 0;
}

Model::~Model() {}

void Model::add(Layer layer) {
    this->layers.push_back(layer);
}

void Model::compile() {
    this->depth = this->layers.size();
    for (unsigned i = 1; i < this->depth; i++) {
        this->layers[i].initWeights(this->layers[i - 1]);
    }
}

void Model::summary() {
    std::cout << "Model of " << this->depth << " layers." << std::endl;
    std::cout << "---------------------" << std::endl << std::endl;
    for (unsigned i; i < this->depth; i++) {
        this->layers[i].summary();
    }
}
