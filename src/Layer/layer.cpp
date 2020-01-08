#include "layer.hpp"

Layer::Layer(unsigned size, Type type, ActivationFunction activation) {
    this->size = size;
    this->type = type;
    this->activation = Activation(activation);
    if (type == Dense) {
        this->neurons = new Matrix<float>(this->size, 1);
    } else {
        this->neurons = NULL;
    }
    this->weights = NULL;
}

Layer::~Layer() {}

void Layer::initWeights(Layer prevLayer) {
    this->weights = new Matrix<float>(this->size, prevLayer.size);
}

std::string Layer::getType() {
    switch(this->type) {
        case Dense:
            return "Dense";
        case Input:
            return "Input";
        default:
            return "Unknown";
    }
}

void Layer::summary() {
    std::cout << this->getType() << " Layer (" << this->size << ")" << std::endl;
    if (this->type != Input) {
        std::cout << "Weights dimension (" << this->weights->getHeight()
                  << ", " << this->weights->getWidth() << ")" << std::endl;
    }
    std::cout << "-------------------------------" << std::endl << std::endl;
}
