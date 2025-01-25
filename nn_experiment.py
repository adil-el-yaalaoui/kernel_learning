import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np
from datasets import SyntheticData
import torch.nn as nn
import torch
import nn_model
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory // 1024 ** 3 - 1  # GPU memory in GB
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8  # Default available RAM in GB



def nn_solution(model,x_train,y_train,x_test,y_test,epochs=20,batch_size=32,print_result=False):
    criterion=nn.MSELoss()
    #criterion=torch.nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_test=TensorDataset(x_test,y_test)
    data_test_loader=DataLoader(dataset_test, batch_size=x_test.shape[0],shuffle=False)

    for epoch in range(epochs):
        model.train()
        for batch_num,full_data in enumerate(dataloader):
            batch_data,batch_labels=full_data
            optimizer.zero_grad()

            pred=model(batch_data).squeeze()
            loss=criterion(pred,batch_labels.float())

            loss.backward()
            optimizer.step()
            class_prediction = (pred >= 0).float() * 2 - 1
            train_accuracy = ((class_prediction == batch_labels).sum())/class_prediction.shape[0]
        if print_result:
            print("Epoch : ",epoch+1, "  /  Loss : ",loss.item(),"  /  Train accuracy : ",train_accuracy.item()*100," %")
            
        model.eval()
        with torch.no_grad():
            batch_data_test,batch_labels_test=next(iter((data_test_loader)))

            pred_test=model(batch_data_test).squeeze()
            test_loss=criterion(pred_test,batch_labels_test.float())

            test_class_pred=(pred_test >= 0).float() * 2 - 1

            test_accuracy=((test_class_pred == batch_labels_test).sum())/test_class_pred.shape[0]

        if print_result:
            print("Epoch : ",epoch+1, "  /  Test Loss : ",test_loss.item(),"  /  Test accuracy : ",test_accuracy.item()*100," %")

    error_classif=1-train_accuracy.item()
    return model,error_classif



def get_experiment_results_separable_nn(model_to_test,noise_levels:list,training_sizes:list,gamma,epochs,batch_size,n_test):
    if model_to_test=="Shallow":
        model=nn_model.NNshallow(50,1)
    elif model_to_test=="Deep":
        model=nn_model.NNdeep(50,1)
    rkhs_norms = {noise: {"NN": []} for noise in noise_levels}
    classification_errors = {noise: {"NN": []} for noise in noise_levels}
    for noise in noise_levels:
        for n_train in training_sizes:
            data=SyntheticData()
            X_train, y_train = data.generate_synthetic_data_separable(n_train, noise=noise)
            X_test, y_test = data.generate_synthetic_data_separable(n_test, noise=noise)
            X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
            X_test = X_test.to(DEVICE)

            # Interpolated solution
              
            shallow_nn,err_classif_nn=nn_solution(model,X_train,y_train,X_test,y_test,epochs,batch_size)
            all_weights = torch.cat([param.view(-1) for param in shallow_nn.parameters()])
            rkhs_norm_nn=all_weights.norm(p=2).detach().numpy()
            rkhs_norms[noise]["NN"].append(rkhs_norm_nn)
            classification_errors[noise]["NN"].append(100 * err_classif_nn)

    return rkhs_norms,classification_errors



#model_nn=shallow_nn_solution(X_train,y_train,X_test,y_test,epochs=20,batch_size=64)
#all_weights = torch.cat([param.view(-1) for param in model_nn.parameters()])
#print(all_weights.norm(p=2))  
