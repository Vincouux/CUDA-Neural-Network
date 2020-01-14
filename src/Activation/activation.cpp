#include "activation.hpp"

Activation::Activation(ActivationFunction activation) {
    this->activationFunction = activation;
    switch (this->activationFunction) {
        case Sigmoid:
            this->activate = [](float x) { return (float)(1.f / (1.f + exp(-1.f * x))); };
            this->derivate = [](float x) { return (float)(exp(-1.f * x) / (1 + exp(-1.f * x))); };
            this->name = "Sigmoid";
            break;
        case Relu:
            this->activate = [](float x) { return x > 0.f ? x : 0.f; };
            this->derivate = [](float x) { return x > 0.f ? 1.f : 0.f; };
            this->name = "Relu";
            break;
        case TanH:
            this->activate = [](float x) { return (float)((1.f - exp(-2.f * x)) / (1.f + exp(-2.f * x))); };
            this->derivate = [](float x) { return (float)(1.f - (((1.f - exp(-2.f * x)) / (1.f + exp(-2.f * x))) * ((1.f - exp(-2.f * x)) / (1.f + exp(-2.f * x))))); };
            this->name = "TanH";
            break;
        case Linear:
            this->activate = [](float x) { return x; };
            this->derivate = [](float x) { (void)x; return 1.f; };
            this->name = "Linear";
            break;
        case LeakyRelu:
            this->activate = [](float x) { return x > 0.f ? x : -0.1f * x; };
            this->derivate = [](float x) { return x > 0.f ? 1.f : -0.1f; };
            this->name = "LeakyRelu";
            break;
    }
}

Activation::~Activation() {}

void Activation::summary() {
    std::cout << "+ Activation " << this->name << std::endl;
}
