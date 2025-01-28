import numpy as np
from sklearn.metrics import accuracy_score

import torch
from eigenpro2.kernels import gaussian
from eigenpro2.models import KernelModel
from sklearn.metrics import accuracy_score
import numpy as np
from datasets import SyntheticData
from nn_experiment import nn_solution
import nn_model

from typing import List, Dict, Tuple, Union


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 ** 3 - 1  # GPU memory in GB
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8  # Default available RAM in GB

# ?
k=300


def compute_gaussian_kernel(X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Computes the Gaussian kernel matrix between two sets of samples.
        X (torch.Tensor): First sample set of shape (n_samples, n_features).
        Y (torch.Tensor): Second sample set of shape (m_samples, n_features).
        gamma (float): Kernel coefficient for Gaussian function.
        """
        return torch.exp(-gamma * torch.cdist(X, Y, p=2) ** 2)


def interpolated_solution(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor, gamma: float) -> Tuple[float, float]:
        """
        Computes the interpolated solution using the kernel matrix and evaluates its performance.
                x_train (torch.Tensor): Training feature set.
                y_train (torch.Tensor): Training labels.
                x_test (torch.Tensor): Testing feature set.
                y_test (torch.Tensor): Testing labels.
                gamma (float): Kernel coefficient for Gaussian function.
        """
        K_train = compute_gaussian_kernel(x_train, x_train, gamma)
        # Solve for alpha = K^-1 y
        alpha_interp = torch.linalg.solve(K_train, y_train)

        # Compute RKHS norm for interpolated solution
        rkhs_norm_interp = torch.sqrt((alpha_interp.T @ (K_train @ alpha_interp)))
        rkhs_norm_interp = rkhs_norm_interp.item()

        # Predict on the test set
        K_test_interp = compute_gaussian_kernel(x_train, x_test, gamma)
        y_pred_interp = torch.sign(K_test_interp.T @ alpha_interp).squeeze()
        error_interp = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_interp.cpu().numpy())

        return rkhs_norm_interp,error_interp


def overfitted_solution(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor, gamma: float, num_classes: int=1, epochs: int = 20, batch_size: int = 64) -> Tuple[float, float]:
        """
        Computes the overfitted solution using a kernel model and evaluates its performance.
                x_train (torch.Tensor): Training feature set.
                y_train (torch.Tensor): Training labels.
                x_test (torch.Tensor): Testing feature set.
                y_test (torch.Tensor): Testing labels.
                gamma (float): Kernel coefficient for Gaussian function.
                epochs (int): Number of training epochs.
                batch_size (int): Batch size for training.
        """        
        kernel_fn = lambda x, y: gaussian(x, y, bandwidth=1.0 / np.sqrt(gamma))
        n_subsamples = min(len(x_train), 5000)
        top_q = min(k, n_subsamples - 1)

        model_overfit = KernelModel(kernel_fn, x_train, num_classes, device=DEVICE)
        model_overfit.predict = lambda samples: model_overfit.forward(samples)

        if num_classes>1:
                y_train_prepared=torch.nn.functional.one_hot(y_train.long(), num_classes=10).float()
                y_test_prepared=torch.nn.functional.one_hot(y_test.long(), num_classes=10).float()
        else:
                y_train_prepared = y_train.unsqueeze(1)
                y_test_prepared = y_test.unsqueeze(1)

        try:
                model_overfit.fit(
                        x_train, y_train_prepared, x_test, y_test_prepared,
                        n_subsamples=n_subsamples, epochs=epochs, mem_gb=DEV_MEM,
                        bs=batch_size, top_q=top_q, print_every=epochs,run_epoch_eval=False)
        except:
                model_overfit.fit(
                        x_train, y_train_prepared, x_test, y_test_prepared,
                        n_subsamples=n_subsamples, epochs=epochs, mem_gb=DEV_MEM,
                        bs=batch_size, print_every=epochs,run_epoch_eval=False)



        #rkhs_norm_overfit = torch.norm(model_overfit.weight).item()
        K_train = kernel_fn(x_train, x_train)  # Compute the full kernel matrix
        alpha_overfit = model_overfit.weight  # Extract learned weights

        # Predict and calculate classification error for overfitted
        if num_classes>1:
                rkhs_norm_overfit = torch.sqrt(torch.sum(alpha_overfit.T @ (K_train @ alpha_overfit))).item()
                y_pred_overfit = model_overfit.predict(x_test).argmax(dim=1)
        else:
                y_pred_overfit = model_overfit.predict(x_test).sign().squeeze()
                rkhs_norm_overfit = torch.sqrt((alpha_overfit.T @ (K_train @ alpha_overfit))).item()
        error_overfit = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_overfit.cpu().numpy())

        return rkhs_norm_overfit,error_overfit


def bayes_solution(x_test: torch.Tensor, y_test: torch.Tensor, threshold: float) -> float:
        """
        Computes the Bayes solution and evaluates its classification error.
        """
        y_pred_bayes = torch.sign(threshold - x_test[:, 0]).to(DEVICE)
        error_bayes = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_bayes.cpu().numpy())
        return error_bayes

def compute_frobenius_norm(model):
    total_norm = 0
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider weight matrices
            total_norm += torch.norm(param, p='fro') ** 2
    return torch.sqrt(total_norm).item()

def compute_spectral_norm(model):
    total_norm = 0
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider weight matrices
            # Compute the largest singular value
            singular_values = torch.linalg.svd(param, full_matrices=False).S
            total_norm += singular_values.max().item()
    return total_norm

def get_experiment_results_separable(model_to_test: str, noise_levels: List[float], training_sizes: List[int], gamma: float, epochs: int, batch_size: int, n_test: int):
        """
        Runs experiments with separable synthetic data and collects results.
                model_to_test (str): Type of neural network model ("Shallow" or "Deep").
                noise_levels (List[float]): Noise levels for the synthetic data.
                training_sizes (List[int]): List of training sample sizes.
                gamma (float): Kernel coefficient for Gaussian function.
                epochs (int): Number of training epochs.
                batch_size (int): Batch size for training.
                n_test (int): Number of test samples.
        """
        if model_to_test=="Shallow":
                model=nn_model.NNshallow(50,1)
        elif model_to_test=="Deep":
                model=nn_model.NNdeep(50,1)
        rkhs_norms = {noise: {"interpolated": [], "overfitted": [],"NN":[]} for noise in noise_levels}
        classification_errors = {noise: {"interpolated": [], "overfitted": [], "NN":[],"bayes": []} for noise in noise_levels}
        for noise in noise_levels:

                for n_train in training_sizes:
                        data=SyntheticData()
                        X_train, y_train = data.generate_synthetic_data_separable(n_train, noise=noise)
                        X_test, y_test = data.generate_synthetic_data_separable(n_test, noise=noise)
                        X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
                        X_test = X_test.to(DEVICE)

                        # Interpolated solution
                        rkhs_norm_interp,error_interp=interpolated_solution(X_train,y_train,X_test,y_test,gamma=gamma)
                        rkhs_norms[noise]["interpolated"].append(rkhs_norm_interp)
                        classification_errors[noise]["interpolated"].append(100 * error_interp)

                        #Overfitted Solution
                        rkhs_norm_overfit,error_overfit=overfitted_solution(X_train,y_train,X_test,y_test,gamma=gamma,epochs=epochs,batch_size=batch_size)
                        rkhs_norms[noise]["overfitted"].append(rkhs_norm_overfit)
                        classification_errors[noise]["overfitted"].append(100 * error_overfit)

                        # Bayes Solution
                        # threshold = 5 for separable data in experience 1
                        error_bayes=bayes_solution(X_test,y_test,threshold=5)
                        classification_errors[noise]["bayes"].append(100 * error_bayes)

                        #Shallow Neural Network Solution
                        shallow_nn,err_classif_nn=nn_solution(model,X_train,y_train,X_test,y_test,epochs,batch_size)
                        #all_weights = torch.cat([param.view(-1) for param in shallow_nn.parameters()])
                        #rkhs_norm_nn=all_weights.norm(p=2).detach().numpy()
                        #rkhs_norms[noise]["NN"].append(rkhs_norm_nn)
                        shallow_frobenius_norm = compute_spectral_norm(shallow_nn)
                        rkhs_norms[noise]["NN"].append(shallow_frobenius_norm)
                        classification_errors[noise]["NN"].append(100 * err_classif_nn)

        return rkhs_norms,classification_errors


def get_experiment_results_non_separable(noise_levels:list,training_sizes:list,gamma,epochs,batch_size,n_test):

      rkhs_norms = {noise: {"interpolated": [], "overfitted": []} for noise in noise_levels}
      classification_errors = {noise: {"interpolated": [], "overfitted": [], "bayes": []} for noise in noise_levels}
      for noise in noise_levels:
        for n_train in training_sizes:
              data=SyntheticData()
              X_train, y_train = data.generate_synthetic_data_non_separable(n_train, noise=noise)
              X_test, y_test = data.generate_synthetic_data_non_separable(n_test, noise=noise)
              X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
              X_test = X_test.to(DEVICE)

              # Interpolated solution
              rkhs_norm_interp,error_interp=interpolated_solution(X_train,y_train,X_test,y_test,gamma=gamma)
              rkhs_norms[noise]["interpolated"].append(rkhs_norm_interp)
              classification_errors[noise]["interpolated"].append(100 * error_interp)

              #Overfitted Solution
              rkhs_norm_overfit,error_overfit=overfitted_solution(X_train,y_train,X_test,y_test,gamma=gamma,epochs=epochs,batch_size=batch_size)
              rkhs_norms[noise]["overfitted"].append(rkhs_norm_overfit)
              classification_errors[noise]["overfitted"].append(100 * error_overfit)

              # Bayes Solution
              # thresghold = 1 for separable data in experience 2 with non separable data
              error_bayes=bayes_solution(X_test,y_test,threshsold=1)
              classification_errors[noise]["bayes"].append(100 * error_bayes)
        
      return rkhs_norms,classification_errors


def interpolated_solution_multiclass(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor, gamma: float, num_classes: int=10) -> Tuple[float, float]:
        """
        Computes the interpolated solution using the kernel matrix and evaluates its performance.
                x_train (torch.Tensor): Training feature set.
                y_train (torch.Tensor): Training labels.
                x_test (torch.Tensor): Testing feature set.
                y_test (torch.Tensor): Testing labels.
                gamma (float): Kernel coefficient for Gaussian function.
        """
        y_train=torch.nn.functional.one_hot(y_train.long(), num_classes=num_classes).float()
        y_test=torch.nn.functional.one_hot(y_test.long(), num_classes=num_classes).float()

        K_train = compute_gaussian_kernel(x_train, x_train, gamma)
        # Solve for alpha = K^-1 y
        alpha_interp = torch.linalg.solve(K_train, y_train)

        # Compute RKHS norm for interpolated solution
        rkhs_norm_interp = torch.sqrt(torch.sum(alpha_interp.T * (K_train @ alpha_interp)))
        rkhs_norm_interp = rkhs_norm_interp.item()

        # Predict on the test set
        K_test_interp = compute_gaussian_kernel(x_train, x_test, gamma)
        y_pred_interp = torch.argmax(K_test_interp.T @ alpha_interp, dim=1)
        y_test_labels = torch.argmax(y_test, dim=1)  # Convert one-hot to class indices
        error_interp = 1 - accuracy_score(y_test_labels.cpu().numpy(), y_pred_interp.cpu().numpy())

        return rkhs_norm_interp,error_interp

def get_experiment_results_mnist(model_to_test: str, noise_levels: List[float], training_sizes: List[int], gamma: float, epochs: int, batch_size: int, n_test: int):
        """
        Runs experiments with MNIST data and collects results.
                model_to_test (str): Type of neural network model ("Shallow" or "Deep").
                noise_levels (List[float]): Noise levels for the synthetic data.
                training_sizes (List[int]): List of training sample sizes.
                gamma (float): Kernel coefficient for Gaussian function.
                epochs (int): Number of training epochs.
                batch_size (int): Batch size for training.
                n_test (int): Number of test samples.
        """
        if model_to_test=="Shallow":
                model=nn_model.NNshallow(784,10)
        elif model_to_test=="Deep":
                model=nn_model.NNdeep(784,10)
        
        rkhs_norms = {noise: {"interpolated": [], "overfitted": [],"NN":[]} for noise in noise_levels}
        classification_errors = {noise: {"interpolated": [], "overfitted": [], "NN":[],"bayes": []} for noise in noise_levels}
        train_loader, test_loader = SyntheticData().load_mnist_data(batch_size=batch_size)

        for noise in noise_levels:
                for n_train in training_sizes:
                        X_train, y_train = [], []
                        for i, (images, labels) in enumerate(train_loader):
                                X_train.append(images.view(images.size(0), -1))
                                y_train.append(labels)
                                if len(X_train) * batch_size >= n_train:
                                        break
                        
                        X_train = torch.cat(X_train)[:n_train].to(DEVICE)
                        y_train = torch.cat(y_train)[:n_train].to(DEVICE).float()

                        #Add noise for training
                        noisy_indices=np.random.choice(n_train, int(noise*n_train), replace=False)
                        print(len(noisy_indices))
                        unique_labels=y_train.unique()
                        y_train_noisy = y_train.clone()
                        for idx in noisy_indices:
                               current_label = y_train_noisy[idx].item()
                               available_labels = unique_labels[unique_labels != current_label]
                               y_train_noisy[idx]=available_labels[torch.randint(len(available_labels), (1,))]

                        # Load all test data (adjust size if necessary)
                        X_test, y_test = [], []
                        for images, labels in test_loader:
                                X_test.append(images.view(images.size(0), -1))
                                y_test.append(labels)
                        X_test = torch.cat(X_test).to(DEVICE)
                        y_test = torch.cat(y_test).to(DEVICE).float()

                        # Interpolated solution
                        rkhs_norm_interp,error_interp=interpolated_solution_multiclass(X_train,y_train_noisy,X_test,y_test,gamma=gamma,num_classes=10)
                        rkhs_norms[noise]["interpolated"].append(rkhs_norm_interp)
                        classification_errors[noise]["interpolated"].append(100 * error_interp)

                        #Overfitted Solution
                        rkhs_norm_overfit,error_overfit=overfitted_solution(X_train,y_train_noisy,X_test,y_test,gamma=gamma,num_classes=10, epochs=epochs,batch_size=batch_size)
                        rkhs_norms[noise]["overfitted"].append(rkhs_norm_overfit)
                        classification_errors[noise]["overfitted"].append(100 * error_overfit)

                        #Shallow Neural Network Solution
                        y_train_one_hot=torch.nn.functional.one_hot(y_train_noisy.long(), num_classes=10).float()
                        y_test_one_hot=torch.nn.functional.one_hot(y_test.long(), num_classes=10).float()
                        shallow_nn,err_classif_nn=nn_solution(model,X_train,y_train_one_hot,X_test,y_test_one_hot,epochs,batch_size)
                        #all_weights = torch.cat([param.view(-1) for param in shallow_nn.parameters()])
                        #rkhs_norm_nn=all_weights.norm(p=2).detach().numpy()
                        #rkhs_norms[noise]["NN"].append(rkhs_norm_nn)
                        shallow_frobenius_norm = compute_spectral_norm(shallow_nn)
                        rkhs_norms[noise]["NN"].append(shallow_frobenius_norm)
                        classification_errors[noise]["NN"].append(100 * err_classif_nn)

        return rkhs_norms,classification_errors