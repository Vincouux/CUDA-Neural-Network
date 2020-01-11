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
    for (unsigned e = 0; e < 10; e++) {

        /* Verbose. */
        std::cout << "Epoch #" << e << std::endl;

        /* Training. */
        for (unsigned i = 0; i < X.getHeight(); i++) {

            /* Verbose. */
            std::cout << i << " / " << X.getHeight() << std::endl;

            /* Copy X to the first layer (input layer). */
            for (unsigned j = 0; j < X.getWidth(); j++) {
                this->layers[0]->getNeurons().setElementAt(j, 0, X.getElementAt(i, j));
            }

            /* Compute the forward pass. */
            this->forward();

            /* Compute the backward pass to propagate the error. */
            this->backward(Y.getLine(i));
        }
    }
}

void Model::forward() {
    for (unsigned j = 1; j < this->depth; j++) {
        Matrix<float> in = this->layers[j]->getWeigths() * this->layers[j - 1]->getNeurons() + this->layers[j]->getBias();
        in.apply(this->layers[j]->getActivation());
        this->layers[j]->setNeurons(in);
    }
}

void Model::backward(const Matrix<float>& Y) {
    Matrix<float> delta = Matrix<float>(0, 0);
    for (int j = this->depth - 1; j >= 0; j--) {
        if ((unsigned)j == this->depth - 1) {
            Matrix<float> tmp = this->layers[j]->getWeigths() * this->layers[j - 1]->getNeurons();
            tmp.apply(this->layers[j]->getActivation());
            delta = (this->layers[j]->getNeurons() - Y) * tmp;
        } else {
            Matrix<float> tmp = this->layers[j]->getWeigths() * this->layers[j - 1]->getNeurons();
            tmp.apply(this->layers[j]->getActivation());
            delta = this->layers[j + 1]->getWeigths() * delta * tmp;
        }
        Matrix<float> deriv = delta * this->layers[j - 1]->getNeurons().transpose();
        this->layers[j]->setWeights(this->layers[j]->getWeigths() - 0.1f * deriv);
    }
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
