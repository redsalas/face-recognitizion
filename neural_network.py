import numpy as np
#import matplotlib.plyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weight1    = np.random.rand(self.input.shape[1], 4)
        self.weight2    = np.random.rand(4, 1)
        self.y          = y
        self.output     = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weight1))
        self.output = sigmoid(np.dot(self.layer1, self.weight2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))
        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == '__main__':
    dataset = pd.read_csv('Churn_Modelling.csv')
    pd.get_dummies(dataset, columns=['Gender','Geography'], drop_first=True)
    dataset.head()

    x = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values



