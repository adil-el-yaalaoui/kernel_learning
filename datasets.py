import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from typing import Tuple


class SyntheticData:
    def __init__(self):
        pass
    

    def generate_synthetic_data_separable(self, n_samples: int = 500, noise: float = 0.1, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a synthetic dataset with linearly separable classes.
            n_samples (int): Number of samples to generate. Defaults to 500.
            noise (float): Proportion of labels to flip as noise. Defaults to 0.1.
            random_state (int): Random seed for reproducibility. Defaults to 42.
        """
        np.random.seed(random_state)

        # Generate binary labels
        y = np.random.choice([-1, 1], size=n_samples)

        # Generate separable features
        x1 = np.where(y == 1, np.random.normal(0, 1, size=n_samples), np.random.normal(10, 1, size=n_samples))
        x_rest = np.random.uniform(-1, 1, size=(n_samples, 49))
        X = np.column_stack((x1, x_rest)).astype(np.float32)

        # Introduce noise by flipping labels
        noise_idx = np.random.choice(n_samples, int(noise * n_samples), replace=False)
        y[noise_idx] *= -1

        return torch.tensor(X), torch.tensor(y).float()
    
    def generate_synthetic_data_non_separable(self, n_samples: int = 500, noise: float = 0.1, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a synthetic dataset with non-linearly separable classes.
            n_samples (int): Number of samples to generate. Defaults to 500.
            noise (float): Proportion of labels to flip as noise. Defaults to 0.1.
            random_state (int): Random seed for reproducibility. Defaults to 42.
            """
        np.random.seed(random_state)

        # Generate binary labels
        y = np.random.choice([-1, 1], size=n_samples)

        # Generate non-separable features
        x1 = np.where(y == 1, np.random.normal(0, 1, size=n_samples), np.random.normal(2, 1, size=n_samples))
        x_rest = np.random.uniform(-1, 1, size=(n_samples, 49))
        X = np.column_stack((x1, x_rest)).astype(np.float32)

        # Introduce noise by flipping labels
        noise_idx = np.random.choice(n_samples, int(noise * n_samples), replace=False)
        y[noise_idx] *= -1

        return torch.tensor(X), torch.tensor(y).float()


    def load_mnist_data(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """
        Loads the MNIST dataset and prepares DataLoaders for training and testing.
            batch_size (int): Number of samples per batch.
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        mnist_train_loader=DataLoader(mnist_train,batch_size,shuffle=True)
        mnist_test_loader=DataLoader(mnist_test,batch_size,shuffle=True)

        return mnist_train_loader,mnist_test_loader
    
    def load_cifar10_data(self,batch_size):
        """
        Loads the CIFAR-10 dataset and prepares DataLoaders for training and testing.
            batch_size (int): Number of samples per batch.
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        cifar_train_loader=DataLoader(cifar_train,batch_size,shuffle=True)
        cifar_test_loader=DataLoader(cifar_test,batch_size,shuffle=True)

        return cifar_train_loader,cifar_test_loader
