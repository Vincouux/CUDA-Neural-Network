#include "Model/model.hpp"
#include "Layer/input.hpp"
#include "Layer/dense.hpp"
#include "Activation/activation.hpp"

int main() {

    bool XOR = false;

    if (XOR) {
        Model model = Model();
        model.add(new Input(2));
        model.add(new Dense(8, TanH));
        model.add(new Dense(1, Sigmoid));
        model.compile();
        model.summary();
        Matrix<float> X = Matrix<float>({{0., 1.}, {1., 0.}, {0., 0.}, {1., 1.}});
        Matrix<float> Y = Matrix<float>({{1.}, {1.}, {0.}, {0.}});
        model.fit(X, Y, 2000, 0.1f, 1, true);
        Matrix<float> P = model.predict(X);
        Y.transpose().display();
        P.transpose().display();
    } else {
        Model model = Model();
        model.add(new Input(784));
        model.add(new Dense(256, TanH));
        model.add(new Dense(10, Sigmoid));
        model.compile();
        model.summary();
        Matrix<float> X = Matrix<float>("data/x_test.csv") / 255.f;
        Matrix<float> Y = Matrix<float>("data/y_test.csv");
        model.fit(X, Y, 1, 0.01f, 1, true);
        Matrix<float> P = model.predict(X);
        for (unsigned i = 0; i < 20; i++) {
            std::cout << "Expect:" << std::endl;
            Y.getLine(i).display();
            std::cout << "Got" << std::endl;
            P.getLine(i).display();
            std::cout << std::endl << "------------------" << std::endl;
        }
    }

    return 0;
}
