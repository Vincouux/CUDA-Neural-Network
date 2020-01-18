# CUDA NEURAL NETWORK

**Simple Neural Network implementation in C++ using Cuda GPU Programming.**

## How it works ?

### Includes
```cpp
#include "Model/model.hpp"
#include "Layer/input.hpp"
#include "Layer/dense.hpp"
#include "Activation/activation.hpp"
```

### Instantiate a new Model
```cpp
Model model = Model();
```
### Add Layers
```cpp
model.add(new Input(784));
model.add(new Dense(256, TanH));
model.add(new Dense(10, Sigmoid));
```

### Compile & Summary
```cpp
model.compile();
model.summary();
```

### Load Data & Fit
```cpp
Matrix<float> X = Matrix<float>("data/x_train.csv") / 255.f;
Matrix<float> Y = Matrix<float>("data/y_train.csv");

/*
Model::fit(const Matrix<float>& X, const Matrix<float>& Y,
           unsigned epochs, float lr, bool earlystopping, bool verbose)
*/
model.fit(X, Y, 1000, 0.001f, true, true);
```

### Predicting & Testing
```cpp
Matrix<float> XT = Matrix<float>("data/x_test.csv") / 255.f;
Matrix<float> YT = Matrix<float>("data/y_test.csv");
Matrix<float> YP = model.predict(XT);
float error = 0.5f * (YT - YP).power(2).sum() / YT.getHeight();
std::cout << "Error: " << error << std::endl;
```

> Work In Progress
