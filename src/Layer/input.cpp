#include "input.hpp"

Input::Input(unsigned size) : Layer(size) {}

void Input::summary() {
    std::cout << "Input Layer " << this->size << std::endl;
    std::cout << "Neurons: " << std::endl;
    this->neurons.display();
    std::cout << "---------------------" << std::endl << std::endl;
}
