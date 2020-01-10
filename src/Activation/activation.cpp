#include "activation.hpp"

Activation::Activation(ActivationFunction activation) {
    this->activationFunction = activation;
    switch (this->activationFunction) {
        case Sigmoid:
            this->activate = [](float x) { return (float)(1.f / (1.f + exp(-1.f * x))); };
            this->derivate = [](float x) { return (float)(exp(-1.f * x) / (1 + exp(-1.f * x))); };
            break;
        case Relu:
            this->activate = [](float x) { return x > 0.f ? x : 0.f; };
            this->derivate = [](float x) { return x > 0.f ? 1.f : 0.f; };
            break;
        case TanH:
            this->activate = [](float x) { return (float)((1.f - exp(-2.f * x)) / (1.f + exp(-2.f * x))); };
            this->derivate = [](float x) { return (float)(1.f - (((1.f - exp(-2.f * x)) / (1.f + exp(-2.f * x))) * ((1.f - exp(-2.f * x)) / (1.f + exp(-2.f * x))))); };
            break;
        case Linear:
            this->activate = [](float x) { return x; };
            this->derivate = [](float x) { (void)x; return 1.f; };
    }
}

Activation::~Activation() {}
