#include "Model/model.hpp"
#include "Layer/input.hpp"
#include "Layer/dense.hpp"
#include "Activation/activation.hpp"

int main() {
    Model model = Model();
    model.add(new Input(3));
    model.add(new Dense(2, Sigmoid));
    model.add(new Dense(1, Sigmoid));
    model.compile();
    model.summary();
    return 0;
}
