#include "model.hpp"

Model::Model() {
    this->layers = std::vector<Layer*>();
    this->depth = 0;
}

Model::~Model() {}

void Model::add(Layer* layer) {
    this->layers.push_back(layer);
}

void Model::fit(const Matrix<float>& X, const Matrix<float>& Y) {
    for (unsigned i = 0; i < X.getHeight(); i++) {

        /* Copy X to the first layer (input layer). */
        for (unsigned j = 0; j < X.getWidth(); j++) {
            this->layers[0]->getNeurons().setElementAt(j, 0, X.getElementAt(i, j));
        }

        /* Compute the forward pass. */
        this->forward();


        /* Compute the error. */
        float error = 0.5f * (this->layers[this->depth - 1]->getNeurons() - Y.getLine(i).transpose()).power(2).sum();
        (void)error;

        /* Compute the backward pass to propagate the error. */
        this->backward();
    }
}

void Model::forward() {
    for (unsigned j = 1; j < this->depth; j++) {
        Matrix<float> in = this->layers[j]->getWeigths() * this->layers[j - 1]->getNeurons() + this->layers[j]->getBias();
        in.apply(this->layers[j]->getActivation());
        this->layers[j]->setNeurons(in);
    }
}

void Model::backward() {
    return;
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
