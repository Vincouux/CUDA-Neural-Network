#include "model.hpp"

Model::Model() {
    this->layers = std::vector<Layer*>();
    this->depth = 0;
}

Model::~Model() {
}

void Model::add(Layer* layer) {
    this->layers.push_back(layer);
}

void Model::fit(const Matrix<float>& X, const Matrix<float>& Y, unsigned epochs, float lr, unsigned batchSize, bool es) {
    double error = 0;
    float last_error = 0;
    unsigned inc = 0;
    for (unsigned e = 0; e < epochs; e++) {

        /* Verbose. */
        std::cout << "Epoch #" << e + 1 << " - Error ";
        error = 0;

        /* Training. */
        for (unsigned i = 0; i < X.getHeight() / batchSize; i++) {

            /* Compute the forward pass by batch. */
            Matrix<float> outputError = 0.f * Matrix<float>(this->layers[this->depth - 1]->getNeurons().getHeight(), this->layers[this->depth - 1]->getNeurons().getWidth());
            for (unsigned k = 0; k < batchSize; k++) {

                /* Copy X to the first layer (input layer). */
                for (unsigned j = 0; j < X.getWidth(); j++) {
                    this->layers[0]->getNeurons().setElementAt(j, 0, X.getElementAt(i * batchSize + k, j));
                }

                /* Forward. */
                this->forward();

                /* Adding the error. */
                outputError = outputError + this->layers[this->depth - 1]->getNeurons() - Y.getLine(i * batchSize + k).transpose();
            }

            outputError = outputError / (float) batchSize;

            /* Compute the backward pass to propagate the error. */
            this->backward(outputError, lr);

            /* Add the error */
            error += outputError.power(2).sum();
        }

        /* Verbose. */
        std::cout << 0.5 * error / (X.getHeight() / batchSize) << std::endl;

        if (es && inc > 10) {
            std::cout << "Early stopping !" << std::endl;
            break;
        }

        if (error > last_error) {
            inc += 1;
        } else {
            inc = 0;
        }

        last_error = error;
    }
}

void Model::forward() {
    for (unsigned j = 1; j < this->depth; j++) {
        Matrix<float> in = this->layers[j]->getWeigths() * this->layers[j - 1]->getNeurons() + this->layers[j]->getBias();
        in.apply(this->layers[j]->getActivation());
        this->layers[j]->setNeurons(in);
    }
}

void Model::backward(const Matrix<float>& error, float lr) {
    Matrix<float> delta = Matrix<float>();
    for (unsigned j = 0; j < this->depth - 1; j++) {
        unsigned i = this->depth - j - 1;
        if (i == this->depth - 1) {
            Matrix<float> tmp = this->layers[i]->getWeigths() * this->layers[i - 1]->getNeurons();
            tmp.apply(this->layers[i]->getDerivation());
            delta = error % tmp;
        } else {
            Matrix<float> tmp = this->layers[i]->getWeigths() * this->layers[i - 1]->getNeurons();
            tmp.apply(this->layers[i]->getDerivation());
            delta = (this->layers[i + 1]->getWeigths().transpose() * delta) % tmp;
        }
        Matrix<float> deriv = delta * this->layers[i - 1]->getNeurons().transpose();
        this->layers[i]->setWeights(this->layers[i]->getWeigths() - lr * deriv);
        this->layers[i]->setBias(this->layers[i]->getBias() - 0.1f * lr * delta);
    }
}

Matrix<float> Model::predict(const Matrix<float>& X) {
    Matrix<float> result = Matrix<float>(X.getHeight(), this->layers[this->depth - 1]->getNeurons().getHeight());
    for (unsigned i = 0; i < X.getHeight(); i++) {

        /* Copy X to the first layer (input layer). */
        for (unsigned j = 0; j < X.getWidth(); j++) {
            this->layers[0]->getNeurons().setElementAt(j, 0, X.getElementAt(i, j));
        }

        /* Compute the forward pass. */
        this->forward();

        /* Add last layer neurons to the result. */
        for (unsigned j = 0; j < this->layers[this->depth - 1]->getNeurons().getHeight(); j++) {
            result.setElementAt(i, j, this->layers[this->depth - 1]->getNeurons().getElementAt(j, 0));
        }
    }
    return result;
}

void Model::compile() {
    this->depth = this->layers.size();
    for (unsigned i = 1; i < this->depth; i++) {
        this->layers[i]->initWeights(this->layers[i - 1]->getSize());
    }
}

void Model::summary() {
    std::cout << "Model of " << this->depth << " layers." << std::endl;
    std::cout << "---------------------" << std::endl << std::endl;
    for (unsigned i = 0; i < this->depth; i++) {
        this->layers[i]->summary();
    }
}
