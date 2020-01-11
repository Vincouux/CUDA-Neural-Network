#include "Model/model.hpp"
#include "Layer/input.hpp"
#include "Layer/dense.hpp"
#include "Activation/activation.hpp"

int main() {
    Model model = Model();
    model.add(new Input(2));
    model.add(new Dense(3, Sigmoid));
    model.add(new Dense(1, Sigmoid));
    model.compile();
    model.summary();

    Matrix<float> X = Matrix<float>({{0., 1.}, {1., 0.}, {1., 1.}});
    Matrix<float> Y = Matrix<float>({{1.}, {1.}, {0.}});

    model.fit(X, Y);

    return 0;
}
