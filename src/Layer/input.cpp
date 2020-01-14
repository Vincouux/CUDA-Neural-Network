#include "input.hpp"

Input::Input(unsigned size) : Layer(size) {}

void Input::summary() {
    std::cout << "+ Layer <Input> of size " << this->size << std::endl;
    std::cout << "+----------------------------" << std::endl;
}
