#include "input.hpp"

Input::Input(unsigned size) : Layer(size) {}

void Input::summary() {
    std::cout << "Input Layer " << this->size << std::endl;
}
