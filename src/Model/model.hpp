#ifndef MODEL_H
#define MODEL_H

class Model {
public:
    Model();
    ~Model();
    void add(Layer layer);
    void compile();

private:
    std::vector<Layer> layers;
    unsigned depth;
};

#endif
