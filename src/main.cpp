#include "Model/model.hpp"
#include "Layer/layer.hpp"
#include "Activation/activation.hpp"

int main() {
    Model model = Model();
    model.add(Layer(3, Input, Sigmoid));
    model.add(Layer(2, Dense, Sigmoid));
    model.add(Layer(1, Dense, Sigmoid));
    model.compile();
    model.summary();
    return 0;
}
