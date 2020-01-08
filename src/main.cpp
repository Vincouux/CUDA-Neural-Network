#include "Model/model.hpp"

int main(int argc, char** argv) {
    Model model = Model();
    model.add(Layer(2, Layer::Input));
    model.add(Layer(5, Layer::Dense, ActivationFunction::Sigmoid));
    model.add(Layer(1, Layer::Dense, ActivationFunction::Sigmoid));
    model.compile();
    return 0;
}
