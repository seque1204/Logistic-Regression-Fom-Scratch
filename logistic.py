"""
Logistic Regression from Scratch
-------------------------------
Implements a numerically stable logistic regression classifier using NumPy.
- Generates synthetic data with sklearn's make_moons
- Trains and evaluates a logistic regression model
- Plots training data and loss curve

Requirements:
- numpy
- matplotlib
- scikit-learn
- pandas

Usage:
    python logistic.py
"""

#Libraries 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

# Numerically stable log(sigmoid(t))
def log_sig(t):
   return np.where(t <= 0, t - np.log(1 + np.exp(t)), -np.log(1 + np.exp(-t)))

# Numerically stable log(1 - sigmoid(t))
def log_one_sig(t):
    result = np.where(t<= 0, -np.log(1+np.exp(t)), -t - np.log(1+np.exp(-t)))
    return result

# Binary cross-entropy loss function
def loss(y, z):
    if isinstance(y, list):
        y = np.array(y)
    y = y.reshape(-1,1)
    z = z.reshape(-1,1)
    loss = -y * log_sig(z) - (1 - y) * log_one_sig(z)
    return loss.mean()

# Linear model: z = Xw + b
def model(w,b,X):
  w.reshape(-1,1)
  z = np.matmul(X,w) + b
  return z.reshape(-1,1)

# Compute gradients for weights and bias
def gradients(X, y, y_hat):
  y = y.reshape(-1,1)
  N = len(y)
  dw = (1/N) * np.matmul(X.T, (y_hat - y))
  db = np.mean(y_hat - y)
  return dw, db

# Training loop for logistic regression
def train(w, b, X, y, iter, lr):
  losses = []
  for _ in range(iter):
    z = model(w, b, X)           # Linear output
    y_hat = sigmoid(z)           # Predicted probabilities
    dw, db = gradients(X, y, y_hat)  # Compute gradients
    w = w - lr * dw              # Update weights
    b = b - lr * db              # Update bias
    current_loss = loss(y, z)    # Compute loss
    losses.append(current_loss)
  return w, b, losses

# Predict class labels (0 or 1) from logits
def predict(z):
  y_hat = np.round(sigmoid(z)).astype(int)
  return y_hat

# Compute accuracy between true and predicted labels
def accuracy(y, y_label):
  y_flat, y_label_flat = y.flatten(), y_label.flatten()
  correct = np.sum(y_label_flat == y_flat)
  total = len(y)
  accuracy = correct/total
  return accuracy

from sklearn.datasets import make_moons

def main():
    acc = acc
    # Generate synthetic training and test data
    X_train, y_train = make_moons(n_samples=500, noise=0.1)
    X_test, y_test = make_moons(n_samples=1000, noise=0.1)

    # Plot training data
    plt.figure()
    plt.plot(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], "g^", label="Class 0")
    plt.plot(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], "bs", label="Class 1")
    plt.legend()
    plt.title("Training Data")

    # Initialize weights and bias
    w = np.random.rand(X_train.shape[1], 1)
    b = 0

    # Train the model
    w, b, loss_seq = train(w, b, X_train, y_train, iter=1000, lr=0.1)

    # Plot loss curve
    plt.figure()
    plt.plot(loss_seq)
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    # Training accuracy
    z = model(w, b, X_train)
    print("Training Accuracy: ", accuracy(np.squeeze(y_train), predict(z)))

    # Test accuracy
    z = model(w, b, X_test)
    y_test = np.squeeze(y_test)
    print("Test Accuracy: ", accuracy(y_test, predict(z)))

    plt.show()



if __name__ == "__main__":
  main()

