import numpy as np
from sklearn.metrics import accuracy_score

import torch
from eigenpro2.kernels import gaussian,ntk_relu
from eigenpro2.models import KernelModel
from sklearn.metrics import accuracy_score
from datasets import SyntheticData
from nn_experiment import nn_solution
import nn_model
import eigenpro2

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 ** 3 - 1  # GPU memory in GB
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8  # Default available RAM in GB

# ?
k=160

def k_ntk(x, xprime):
    with torch.no_grad():
        v = torch.linalg.norm(x) * torch.linalg.norm(xprime)
        u = .99999 * torch.dot(x, xprime) / v
        return v * (u * (torch.pi - torch.arccos(u) + torch.sqrt(1 - u ** 2) )/ (2 * np.pi)
                    +  u * (torch.pi - torch.arccos(u)) /  (2 * np.pi))

def ntk_kernel(x,z):
    n,_=x.shape
    m,_=z.shape
    H = torch.empty((n, m))
    for i in range(n):
        for j in range(m):
            H[i,j] = k_ntk(x[i], z[j])

    return H

def kappa(u,v):
    u=.99999*u
    return v * (u * (torch.pi - torch.arccos(u) + torch.sqrt(1 - u ** 2) )/ (2 * np.pi)
                    +  u * (torch.pi - torch.arccos(u)) /  (2 * np.pi))

def kappa2(u):
    u=.99999*u
    return 2*u/torch.pi * (torch.pi - torch.arccos(u))  + torch.sqrt(1 - u ** 2) /torch.pi

def easier_ntk(x,z):
    inner_prod=x@z.T
    norm_x=x.norm(dim=-1)
    norm_z=z.norm(dim=-1)
    norm_mat=norm_x.unsqueeze(1)@norm_z.unsqueeze(1).T

    return kappa(inner_prod/norm_mat,norm_mat)

def easier_ntk2(x,z):
    inner_prod=x@z.T
    norm_x=x.norm(dim=-1)
    norm_z=z.norm(dim=-1)
    norm_mat=norm_x.unsqueeze(1)@norm_z.unsqueeze(1).T

    return norm_mat*kappa2(inner_prod/norm_mat)



def interpolated_solution_ntk(x_train,y_train,x_test,y_test):
        
        K_train = easier_ntk2(x_train, x_train)
        # Solve for alpha = K^-1 y
        alpha_interp = torch.linalg.solve(K_train, y_train)

        # Compute RKHS norm for interpolated solution
        rkhs_norm_interp = torch.sqrt((alpha_interp @ (K_train @ alpha_interp)))
        rkhs_norm_interp = rkhs_norm_interp.item()

        # Predict on the test set
        K_test_interp = easier_ntk2(x_train, x_test)
        y_pred_interp = torch.sign(K_test_interp.T @ alpha_interp).squeeze()
        error_interp = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_interp.cpu().numpy())

        return rkhs_norm_interp,error_interp

def overfitter_ntk(X_train,y_train,X_test,y_test,epochs,batch_size):
        kernel_fn = lambda x, y: easier_ntk2(x, y)

        model = eigenpro2.KernelModel(kernel_fn, X_train, 1, device=torch.device("cpu"))
        n_subsamples = min(len(X_train), 5000)
        top_q = min(160, n_subsamples - 1)

        #results = model.fit(X_train, y_train.unsqueeze(1), X_test, y_test.unsqueeze(1), epochs=20, print_every=2, mem_gb=8,top_q=top_q)
        try:
             result_overfit = model.fit(
                 X_train, y_train.unsqueeze(1), X_test, y_test.unsqueeze(1),
                 n_subsamples=n_subsamples, epochs=epochs, mem_gb=DEV_MEM,
                 bs=batch_size, top_q=top_q, print_every=epochs,run_epoch_eval=False)
        except:
              result_overfit = model.fit(
                        X_train, y_train.unsqueeze(1), X_test, y_test.unsqueeze(1),
                        n_subsamples=n_subsamples, epochs=epochs, mem_gb=DEV_MEM,
                        bs=batch_size, print_every=epochs,run_epoch_eval=False)
              
        coeff_kernel=model.weight.squeeze() 
        kernel_train=model.kernel_matrix(X_train)

        rkhs_norm_overfit = torch.sqrt(coeff_kernel@(kernel_train@coeff_kernel))

        # Predict and calculate classification error for overfitted
        y_pred_overfit = model.forward(X_test).sign().squeeze()
        error_overfit = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_overfit.cpu().numpy())

        return rkhs_norm_overfit,error_overfit


def compute_gaussian_kernel(X, Y, gamma):
        return torch.exp(-gamma * torch.cdist(X, Y, p=2) ** 2)


def interpolated_solution(x_train,y_train,x_test,y_test,gamma):
        
        K_train = compute_gaussian_kernel(x_train, x_train, gamma)
        # Solve for alpha = K^-1 y
        alpha_interp = torch.linalg.solve(K_train, y_train)

        # Compute RKHS norm for interpolated solution
        rkhs_norm_interp = torch.sqrt((alpha_interp @ (K_train @ alpha_interp)))
        rkhs_norm_interp = rkhs_norm_interp.item()

        # Predict on the test set
        K_test_interp = compute_gaussian_kernel(x_train, x_test, gamma)
        y_pred_interp = torch.sign(K_test_interp.T @ alpha_interp).squeeze()
        error_interp = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_interp.cpu().numpy())

        return rkhs_norm_interp,error_interp

def overfitted_solution(x_train,y_train,x_test,y_test,gamma,epochs=20,batch_size=64):
        
        kernel_fn = lambda x, y: gaussian(x, y, bandwidth=1.0 / np.sqrt(gamma))
        n_subsamples = min(len(x_train), 5000)
        top_q = min(k, n_subsamples - 1)

        model_overfit = KernelModel(kernel_fn, x_train, 1, device=DEVICE)
        model_overfit.predict = lambda samples: model_overfit.forward(samples)

        try:
             result_overfit = model_overfit.fit(
                 x_train, y_train.unsqueeze(1), x_test, y_test.unsqueeze(1),
                 n_subsamples=n_subsamples, epochs=epochs, mem_gb=DEV_MEM,
                 bs=batch_size, top_q=top_q, print_every=epochs,run_epoch_eval=False)
        except:
              result_overfit = model_overfit.fit(
                        x_train, y_train.unsqueeze(1), x_test, y_test.unsqueeze(1),
                        n_subsamples=n_subsamples, epochs=epochs, mem_gb=DEV_MEM,
                        bs=batch_size, print_every=epochs,run_epoch_eval=False)
                
        rkhs_norm_overfit = torch.norm(model_overfit.weight).item()

        # Predict and calculate classification error for overfitted
        y_pred_overfit = model_overfit.predict(x_test).sign().squeeze()
        error_overfit = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_overfit.cpu().numpy())


        return rkhs_norm_overfit,error_overfit


def bayes_solution(x_test,y_test,threshsold):
        y_pred_bayes = torch.sign(threshsold - x_test[:, 0]).to(DEVICE)
        error_bayes = 1 - accuracy_score(y_test.cpu().numpy(), y_pred_bayes.cpu().numpy())
        return error_bayes


def get_experiment_results_separable(noise_levels:list,training_sizes:list,gamma,epochs,batch_size,n_test):
        rkhs_norms = {noise: {"interpolated": [], "overfitted": []} for noise in noise_levels}
        classification_errors = {noise: {"interpolated": [], "overfitted": [],"bayes": []} for noise in noise_levels}
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
                        # thresghold = 5 for separable data in experience 1
                        error_bayes=bayes_solution(X_test,y_test,threshsold=5)
                        
                        classification_errors[noise]["bayes"].append(100 * error_bayes)

        return rkhs_norms,classification_errors


def get_experiment_results_separable_ntk(noise_levels:list,training_sizes:list,epochs,batch_size,n_test):
        rkhs_norms = {noise: {"interpolated": [], "overfitted": []} for noise in noise_levels}
        classification_errors = {noise: {"interpolated": [], "overfitted": [],"bayes": []} for noise in noise_levels}
        for noise in noise_levels:
                for n_train in training_sizes:
                        data=SyntheticData()
                        X_train, y_train = data.generate_synthetic_data_separable(n_train, noise=noise)
                        X_test, y_test = data.generate_synthetic_data_separable(n_test, noise=noise)
                        X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
                        X_test = X_test.to(DEVICE)

                        # Interpolated solution
                        rkhs_norm_interp,error_interp=interpolated_solution_ntk(X_train,y_train,X_test,y_test)
                        rkhs_norms[noise]["interpolated"].append(rkhs_norm_interp)
                        classification_errors[noise]["interpolated"].append(100 * error_interp)

                       #Overfitted Solution
                        rkhs_norm_overfit,error_overfit=overfitter_ntk(X_train,y_train,X_test,y_test,epochs=epochs,batch_size=batch_size)
                        rkhs_norms[noise]["overfitted"].append(rkhs_norm_overfit)
                        classification_errors[noise]["overfitted"].append(100 * error_overfit)

                        # Bayes Solution
                        # thresghold = 5 for separable data in experience 1
                        error_bayes=bayes_solution(X_test,y_test,threshsold=5)
                        classification_errors[noise]["bayes"].append(100 * error_bayes)

        return rkhs_norms,classification_errors


def get_experiment_results_non_separable_ntk(noise_levels:list,training_sizes:list,epochs,batch_size,n_test):
        rkhs_norms = {noise: {"interpolated": [], "overfitted": []} for noise in noise_levels}
        classification_errors = {noise: {"interpolated": [], "overfitted": [],"bayes": []} for noise in noise_levels}
        for noise in noise_levels:
                for n_train in training_sizes:
                        data=SyntheticData()
                        X_train, y_train = data.generate_synthetic_data_non_separable(n_train, noise=noise)
                        X_test, y_test = data.generate_synthetic_data_non_separable(n_test, noise=noise)
                        X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
                        X_test = X_test.to(DEVICE)

                        # Interpolated solution
                        rkhs_norm_interp,error_interp=interpolated_solution_ntk(X_train,y_train,X_test,y_test)
                        rkhs_norms[noise]["interpolated"].append(rkhs_norm_interp)
                        classification_errors[noise]["interpolated"].append(100 * error_interp)
                        print("Overfit Begin ! ")
                       #Overfitted Solution
                        rkhs_norm_overfit,error_overfit=overfitter_ntk(X_train,y_train,X_test,y_test,epochs=epochs,batch_size=batch_size)
                        rkhs_norms[noise]["overfitted"].append(rkhs_norm_overfit)
                        classification_errors[noise]["overfitted"].append(100 * error_overfit)

                        print("Overfit Over ! ")

                        # Bayes Solution
                        # thresghold = 5 for separable data in experience 1
                        error_bayes=bayes_solution(X_test,y_test,threshsold=1)
                        classification_errors[noise]["bayes"].append(100 * error_bayes)

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
      



        
        

        

