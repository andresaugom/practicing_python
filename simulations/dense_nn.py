import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=500, noise=0.2, random_state=40)

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 10)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.output(x)
        return x
