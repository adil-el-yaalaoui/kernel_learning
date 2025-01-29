from datasets import SyntheticData
from experiments import *
from figure_plot import plot_solutions

import torch
import matplotlib.pyplot as plt

training_sizes = [200, 1001, 2000, 5000, 10000]
n_test = 1000
noise_levels = [0.0, 0.01, 0.1]   # 0%, 1%, 10% noise
gamma = 0.1  # Kernel bandwidth
epochs = 10
batch_size=1024
reg_lambda = 0.1  # Regularization parameter for overfitting
fig = plt.figure(figsize=(14, 12))

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 ** 3 - 1  # GPU memory in GB
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8  # Default available RAM in GB


kernel_to_test="NTK"
data_to_test="separable"
rkhs_norms,classification_errors=get_experiment_results(kernel_to_test,noise_levels,training_sizes,epochs,batch_size,n_test,data_to_test)
plot_solutions(fig,noise_levels,training_sizes,classification_errors,rkhs_norms,"NTK_results")

