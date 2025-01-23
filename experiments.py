import numpy as np
from sklearn.metrics import accuracy_score

import torch
from eigenpro2.kernels import gaussian
from eigenpro2.models import KernelModel
from sklearn.metrics import accuracy_score
import numpy as np
from datasets import SyntheticData

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 ** 3 - 1  # GPU memory in GB
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8  # Default available RAM in GB

# ?
k=160


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
      classification_errors = {noise: {"interpolated": [], "overfitted": [], "bayes": []} for noise in noise_levels}
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
      



        
        

        

