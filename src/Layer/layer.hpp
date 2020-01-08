#ifndef LAYER_H
#define LAYER_H

class Layer {
public:
    Layer(size_t size, Type type, Activation activation=ActivationFunction::Linear);
    ~Layer();
    enum Type {
        Input,
        Dense
    };
    void initWeights(Layer prevLayer);

private:
    size_t size;
    Activation activation;
    Type type;
};

#endif
