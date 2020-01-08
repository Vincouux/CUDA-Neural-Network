#ifndef INPUT_H
#define INPUT_H

#include <string>
#include "layer.hpp"

class Input: public Layer {
public:
    Input(unsigned size);
    void initWeights(unsigned prevSize) { (void)prevSize; }
    void summary();
};

#endif
