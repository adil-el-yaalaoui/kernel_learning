import numpy as np
from sklearn.metrics import accuracy_score

import torch
from eigenpro2.kernels import gaussian,ntk_relu
from eigenpro2.models import KernelModel
from sklearn.metrics import accuracy_score
from datasets import SyntheticData
import eigenpro2
import gc

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 ** 3 - 1  # GPU memory in GB
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8  # Default available RAM in GB

# ?
k=160


# Kappa formula from the TP
def kappa(u,v):
    u=.99999*u
    return v * (u * (torch.pi - torch.arccos(u) + torch.sqrt(1 - u ** 2) )/ (2 * np.pi)
                    +  u * (torch.pi - torch.arccos(u)) /  (2 * np.pi))

# Kappa formula from the course
def kappa2(u):
    u=.99999*u
    return 2*u/torch.pi * (torch.pi - torch.arccos(u))  + torch.sqrt(1 - u ** 2) /torch.pi

# Building the kernel matrix with Kappa
def easier_ntk(x,z):
    inner_prod=x@z.T
    norm_x=x.norm(dim=-1)
    norm_z=z.norm(dim=-1)
    norm_mat=norm_x.unsqueeze(1)@norm_z.unsqueeze(1).T

    return kappa(inner_prod/norm_mat,norm_mat)

# Building the kernel matrix from the Kappa2 function
def easier_ntk2(x,z):
    inner_prod=x@z.T
    norm_x=x.norm(dim=-1)
    norm_z=z.norm(dim=-1)
    norm_mat=norm_x.unsqueeze(1)@norm_z.unsqueeze(1).T

    return norm_mat*kappa2(inner_prod/norm_mat)

    
def gaussian_kernel(X, Y, gamma=0.1):
        return gaussian(X,Y,bandwidth=1/np.sqrt(gamma))

def interpolated_solution(kernel,x_train,y_train,x_test,y_test):
        """
        Computes the interpolated solution with the Gaussian Kernel by solving the equation Ka=y
        """
        K_train = kernel(x_train, x_train)
        # Solve for alpha = K^-1 y
        alpha_interp = torch.linalg.solve(K_train, y_train)

        # Compute RKHS norm for interpolated solution
        rkhs_norm_interp = torch.sqrt((alpha_interp @ (K_train @ alpha_interp)))
        rkhs_norm_interp = rkhs_norm_interp.item()

        # Predict on the test set
        K_test_interp = kernel(x_train, x_test)
        y_pred_interp = torch.sign(K_test_interp.T @ alpha_interp).squeeze()
        error_interp = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_interp.cpu().numpy())

        return rkhs_norm_interp,error_interp

def overfitted_solution(kernel,x_train,y_train,x_test,y_test,epochs=20,batch_size=64,data_type="separable"):
        """
        Computes the overfitted solution by applying the EigenPro-SGD algorithm to accelerate gradient descent
        """

        kernel_fn = lambda x, y: kernel(x, y)
        n_subsamples = min(len(x_train), 5000)
        top_q = min(160, n_subsamples - 1)

        model = eigenpro2.KernelModel(kernel_fn, x_train, 1, device=torch.device("cpu"))
        gc.collect()
        if data_type=="separable":
             result_overfit = model.fit(
                 x_train, y_train.unsqueeze(1), x_test, y_test.unsqueeze(1),
                epochs=epochs, mem_gb=DEV_MEM,
                 bs=batch_size,run_epoch_eval=False)
        else:
              if n_subsamples<1000:
                result_overfit = model.fit(
                 x_train, y_train.unsqueeze(1), x_test, y_test.unsqueeze(1),
                epochs=epochs, mem_gb=DEV_MEM,
                 bs=batch_size,run_epoch_eval=False)
              else:
                result_overfit = model.fit(
                        x_train, y_train.unsqueeze(1), x_test, y_test.unsqueeze(1),
                        n_subsamples=n_subsamples, epochs=epochs, mem_gb=DEV_MEM,
                        bs=batch_size, print_every=epochs,run_epoch_eval=False,top_q=top_q)
                
        gc.collect()

        coeff_kernel=model.weight.squeeze() 
        kernel_train=model.kernel_matrix(x_train)
        rkhs_norm_overfit = torch.sqrt(coeff_kernel@(kernel_train@coeff_kernel))

        del x_train,y_train
        del kernel_train
        # Predict and calculate classification error for overfitted
        y_pred_overfit = model.forward(x_test).sign().squeeze()
        error_overfit = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_overfit.cpu().numpy())
        del model

        return rkhs_norm_overfit,error_overfit


def bayes_solution(x_test,y_test,threshsold):
        """
        Computes the Bayes optimal classifier for the separable and non separable data.
        Threshold controls which one to apply.

        Bayes does not exist for other types of datas tested in the experiments.        
        """
        y_pred_bayes = torch.sign(threshsold - x_test[:, 0]).to(DEVICE)
        error_bayes = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_bayes.cpu().numpy())
        return error_bayes



def get_experiment_results(kernel,noise_levels:list,training_sizes:list,epochs,batch_size,n_test,data_type):
        """
        Computes the full results given the nois levels and traning sizes for the Neural Tangent Kernel        
        """
        if kernel=="Gaussian":
              kernel_fn=gaussian_kernel
        elif kernel=="NTK":
              kernel_fn=easier_ntk2
        else:
              raise TypeError("Kernel not valid")
        
        data=SyntheticData()
        rkhs_norms = {noise: {"interpolated": [], "overfitted": []} for noise in noise_levels}
        classification_errors = {noise: {"interpolated": [], "overfitted": [],"bayes": []} for noise in noise_levels}
        for noise in noise_levels:
                #We generate the same data for one noise, and we take the number of training data we want
                # This makes it faster
                if data_type=="separable":
                        X_train, y_train = data.generate_synthetic_data_separable(training_sizes[-1], noise=noise)
                        X_test, y_test = data.generate_synthetic_data_separable(n_test, noise=noise)
                elif data_type=="non-separable":
                        X_train, y_train = data.generate_synthetic_data_non_separable(training_sizes[-1], noise=noise)
                        X_test, y_test = data.generate_synthetic_data_non_separable(n_test, noise=noise)
                        
                else :
                        raise TypeError("Data type not valid")
                
                for n_train in training_sizes:

                        X_train_new, y_train_new = X_train[:n_train].to(DEVICE), y_train[:n_train].to(DEVICE)
                        X_test = X_test.to(DEVICE)

                        # Interpolated solution
                        rkhs_norm_interp,error_interp=interpolated_solution(kernel_fn,X_train_new,y_train_new,X_test,y_test)
                        rkhs_norms[noise]["interpolated"].append(rkhs_norm_interp)
                        classification_errors[noise]["interpolated"].append(100 * error_interp)

                       #Overfitted Solution
                        rkhs_norm_overfit,error_overfit=overfitted_solution(kernel_fn,X_train_new,y_train_new,X_test,y_test,epochs=epochs,batch_size=batch_size,data_type=data_type)
                        rkhs_norms[noise]["overfitted"].append(rkhs_norm_overfit)
                        classification_errors[noise]["overfitted"].append(100 * error_overfit)

                        #Bayes solution
                        if data_type=="separable":
                                error_bayes=bayes_solution(X_test,y_test,threshsold=5)
                                classification_errors[noise]["bayes"].append(100 * error_bayes)
                        else:
                              error_bayes=bayes_solution(X_test,y_test,threshsold=1)
                              classification_errors[noise]["bayes"].append(100 * error_bayes)

                del X_train,y_train,X_test,y_test

        return rkhs_norms,classification_errors

