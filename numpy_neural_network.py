# Import necessary libraries 
import numpy as np
import sklearn.datasets
import matplotlib
import matplotlib.pyplot as plt
import time
from IPython import display
from sklearn.preprocessing import OneHotEncoder

# Change default figure size
matplotlib.rcParams['figure.figsize'] = (14.0, 8.0)

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_blobs(200, centers=3)
X = np.c_[np.ones(X.shape[0]), X]  # Add learnable bias

y_one_hot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

plt.scatter(X[:, 1], X[:, 2], s=40, c=y, cmap=plt.cm.Spectral)

# Generate a decision boundary plot for a classifier's predictions
def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5

    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx_bias = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]

    Z = pred_func(xx_bias)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral)
    plt.show()

# Define neural network class
class NeuralNetwork:
    def __init__(self, input_size, output_size, learning_rate = 0.01):
        self.weights = 2 *np.random.random((input_size, output_size)) - 1
        self.bias = np.random.rand(1, output_size)
        self.learning_rate = learning_rate

    def activation(self, x):
        return np.argmax(x, axis=1)

    def predict(self, x):
        self.output = np.dot(x, self.weights) + self.bias
        return self.activation(self.output)

    def update_weights(self, x, y):
        predictions = self.predict(x)
        y_one_hot_pred = OneHotEncoder(sparse_output=False).fit_transform(predictions.reshape(-1, 1))
        diff = y_one_hot - y_one_hot_pred
        self.weights += self.learning_rate * np.dot(x.T, diff)
        self.bias += self.learning_rate * np.sum(diff, axis=0)

    def one_hot_encode(self, predictions):
        encoder = OneHotEncoder(sparse_output=False)
        return encoder.fit_transform(predictions.reshape(-1, 1))
    
    def train_model(self, X, y, epochs=50, early_stopping_counts=20):
        y_one_hot = self.one_hot_encode(y)
        best_error = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            predictions = self.predict(X)
            error = np.mean(np.abs(y_one_hot - self.one_hot_encode(predictions)))

            if error < best_error:
                best_error = error
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_counts:
                print(f"Early stopping at epoch: {epoch} with error: {error}.")
                break

            display.clear_output(wait=True)
            plot_decision_boundary(lambda x: self.predict(x))
            display.display(f"Epoch: {epoch + 1} / {epochs}, error: {error}.")
            time.sleep(0.5)

            self.update_weights(X, y_one_hot)

# Initialize and train the neural network
neural_network = NeuralNetwork(input_size=X.shape[1], output_size=3, learning_rate=0.01)

neural_network.train_model(X, y)