import numpy as np
from torchvision import datasets, transforms
import torch


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
    
    def load_mnist_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Flatten images and convert to numpy arrays
        X_train = mnist_train.data.numpy().reshape(-1, 28 * 28) / 255.0  # Scale pixel values
        y_train = mnist_train.targets.numpy()
        X_test = mnist_test.data.numpy().reshape(-1, 28 * 28) / 255.0
        y_test = mnist_test.targets.numpy()

        # Normalize features
        X_train, X_test = (X_train - 0.5) / 0.5, (X_test - 0.5) / 0.5  # Normalize to [-1, 1]
        return X_train, y_train, X_test, y_test
    
    def load_cifar10_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Flatten images and convert to numpy arrays
        X_train = cifar_train.data.numpy().reshape(-1, cifar_train.data.shape[1]*cifar_train.data.shape[2]) / 255.0  # Scale pixel values
        y_train = cifar_train.targets.numpy()
        X_test = cifar_test.data.numpy().reshape(-1, cifar_test.data.shape[1]*cifar_test.data.shape[2]) / 255.0
        y_test = cifar_test.targets.numpy()

        # Normalize features
        X_train, X_test = (X_train - 0.5) / 0.5, (X_test - 0.5) / 0.5  # Normalize to [-1, 1]
        return X_train, y_train, X_test, y_test
