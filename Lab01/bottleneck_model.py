import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class autoencoderMLP4Layer_bottleneck(nn.Module):

    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer_bottleneck, self).__init__()

        N2 = 392
        self.fc1 = nn.Linear(N_input, N2)
        self.fc2 = nn.Linear(N2, N_bottleneck)
        self.fc3 = nn.Linear(N_bottleneck, N2)
        self.fc4 = nn.Linear(N2, N_output)
        self.type = 'MLP4'
        self.input_shape = (1, 28*28)

    def encode(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        return X

    def decode(self, X):
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = torch.sigmoid(X)
        return X

    def forward(self, X1, X2, steps):
        # create 2 bottleneck tensors
        bottleneck1 = self.encode(X1)
        bottleneck2 = self.encode(X2)
        # linearly interpolate between them for n steps
        output = []
        for i in range(0, steps):
            weight = i / (steps - 1)
            tensor = torch.lerp(bottleneck1, bottleneck2, weight)
            output.append(self.decode(tensor))

        return output
