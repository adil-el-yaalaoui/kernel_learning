from datasets import SyntheticData
from experiments import *
from figure_plot import plot_solutions, plot_solutions_nn
from nn_experiment import *
import torch
import matplotlib.pyplot as plt

training_sizes = [200, 1001, 2000, 5000, 10000, 30000]
n_test = 1000
noise_levels = [0.0, 0.01, 0.1]  # 0%, 1%, 10% noise
gamma = 0.1  # Kernel bandwidth
epochs = 10
batch_size=256
reg_lambda = 0.1  # Regularization parameter for overfitting
fig = plt.figure(figsize=(14, 12))

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 ** 3 - 1  # GPU memory in GB
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8  # Default available RAM in GB


# Uncomment this if you want to observe the results only for shallow NN
#model_to_test="Deep"
#rkhs_norms,classification_errors=get_experiment_results_separable_nn(model_to_test,noise_levels,training_sizes,gamma,epochs,batch_size,n_test)
#plot_solutions_nn(fig,noise_levels,training_sizes,classification_errors,rkhs_norms)


# Uncomment this if you want to see the results for all models including shallow NN
#model_to_test="Deep"
#rkhs_norms,classification_errors=get_experiment_results_separable(model_to_test, noise_levels,training_sizes,gamma,epochs,batch_size,n_test)
#plot_solutions(fig,noise_levels,training_sizes,classification_errors,rkhs_norms)

#print(rkhs_norms)



####################### MNIST TESTS #######################

training_sizes = [2000, 5000]
noise_levels = [0.01, 0.1]  # 0%, 1%, 10% noise
gamma = 0.01  # Kernel bandwidth
epochs = 10
batch_size=256

rkhs_norms,classification_errors=get_experiment_results_mnist("Deep",noise_levels,training_sizes,gamma,epochs,batch_size,n_test)
plot_solutions(fig,noise_levels,training_sizes,classification_errors,rkhs_norms, bayes=False)
print(rkhs_norms)