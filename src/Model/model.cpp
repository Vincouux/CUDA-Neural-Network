#include "model.hpp"

Model::Model() {
    this->layers = std::vector<Layer*>();
    this->depth = 0;
}

Model::~Model() {}

void Model::add(Layer* layer) {
    this->layers.push_back(layer);
}

void Model::fit(Matrix<float> X, Matrix<float> Y) {
    for (unsigned i = 0; i < X.getHeight(); i++) {
        for (unsigned j = 0; j < X.getWidth(); j++) {
            this->layers[0]->getNeurons().setElementAt(j, 0, X.getElementAt(i, j));
        }
        for (unsigned j = 1; j < this->depth; j++) {
            this->layers[0]->setNeurons(this->layers[j]->getNeurons() * this->layers[j - 1]->getNeurons());
        }
    }
    (void)Y;
}

void Model::compile() {
    this->depth = this->layers.size();
    for (unsigned i = 1; i < this->depth; i++) {
        this->layers[i]->initWeights(this->layers[i - 1]->getSize());
    }
}

void Model::summary() {
    std::cout << "Model of " << this->depth << " layers." << std::endl;
    std::cout << "---------------------" << std::endl << std::endl;
    for (unsigned i = 0; i < this->depth; i++) {
        this->layers[i]->summary();
    }
}
