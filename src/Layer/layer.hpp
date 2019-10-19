#ifndef LAYER_H
#define LAYER_H

class Layer {
public:
  Layer(int size, Activation activation);
  ~Layer();

private:
  Int size;
  Activation activation;
};

#endif
