import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch, os
from torchvision.datasets import MNIST
from torch.nn.functional import one_hot


def unit_range_normalize(samples):
    samples -= samples.min(dim=0, keepdim=True).values
    return samples/samples.max(dim=1, keepdim=True).values


class SyntheticData:
    def __init__(self):
        pass
    

    def generate_synthetic_data_separable(self,n_samples=500, noise=0.1, random_state=42):
        np.random.seed(random_state)

        y = np.random.choice([-1, 1], size=n_samples)

        x1 = np.where(y == 1, np.random.normal(0, 1, size=n_samples), np.random.normal(10, 1, size=n_samples))

        x_rest = np.random.uniform(-1, 1, size=(n_samples, 49))

        X = np.column_stack((x1, x_rest)).astype(np.float32)

        noise_idx = np.random.choice(n_samples, int(noise * n_samples), replace=False)

        y[noise_idx] *= -1

        return torch.tensor(X), torch.tensor(y).float()
    
    def generate_synthetic_data_non_separable(self,n_samples=500, noise=0.1, random_state=42):
        np.random.seed(random_state)

        y = np.random.choice([-1, 1], size=n_samples)

        x1 = np.where(y == 1, np.random.normal(0, 1, size=n_samples), np.random.normal(2, 1, size=n_samples))

        x_rest = np.random.uniform(-1, 1, size=(n_samples, 49))

        X = np.column_stack((x1, x_rest)).astype(np.float32)

        noise_idx = np.random.choice(n_samples, int(noise * n_samples), replace=False)

        y[noise_idx] *= -1

        return torch.tensor(X), torch.tensor(y).float()
    
    def load_mnist_data(self,noise,DEVICE=torch.device("cpu")):
        mnist_train = datasets.MNIST(root='./data', train=True, download=True)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True)

        x_train,y_train=mnist_train.data,mnist_train.targets
        x_test,y_test=mnist_test.data,mnist_test.targets
    

        # Flatten the data
        x_train = x_train.reshape(x_train.shape[0], -1).to(DEVICE).float()
        x_test = x_test.reshape(x_test.shape[0], -1).to(DEVICE).float()

        # Normalize the data
        x_train = unit_range_normalize(x_train)
        x_test = unit_range_normalize(x_test)
        

        # Random permutation of labels with noise
        num_samples_train = len(y_train)
        num_to_permute_train = int(noise * num_samples_train)
    
        indices_to_permute = torch.randperm(num_samples_train)[:num_to_permute_train]
        permuted_indices = indices_to_permute[torch.randperm(num_to_permute_train)]
        y_permuted = y_train.clone()
        y_permuted[indices_to_permute]=y_train[permuted_indices]

        y_permuted=one_hot(y_permuted,num_classes=10).float()
        y_test=one_hot(y_test,num_classes=10).float()

        return x_train,y_permuted,x_test,y_test
    

 


