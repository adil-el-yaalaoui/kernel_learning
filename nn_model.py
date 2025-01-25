import torch.nn as nn
import torch.functional as F


class NNshallow(nn.Module): 
    def __init__(self,input_dim,output_dim, hidden_dim=10):
        super().__init__()
        self.net_type = 'shallow' 
        self.hidden_dim = hidden_dim
        self.in_dim=input_dim
        self.out_dim=output_dim
        self.l1=nn.Linear(self.in_dim,hidden_dim)
        self.lout=nn.Linear(hidden_dim,output_dim)
        self.activ=nn.ReLU()
            
    def forward(self, x):
        x = self.activ(self.l1(x))
        x = self.lout(x)
        return x

class NNdeep(nn.Module):
    def __init__(self,input_dim,output_dim, hidden_dim=10):
        super().__init__()
        self.net_type = 'deep' 
        self.hidden_dim = hidden_dim
        self.in_dim=input_dim
        self.out_dim=output_dim
        self.l1=nn.Linear(self.in_dim,hidden_dim)
        self.l2=nn.Linear(hidden_dim,hidden_dim)
        self.l3=nn.Linear(hidden_dim,hidden_dim)
        self.l4=nn.Linear(hidden_dim,hidden_dim)
        self.lout=nn.Linear(hidden_dim,self.out_dim)

        
    def forward(self, x):
        x = nn.ReLU(self.l1(x))
        x = nn.ReLU(self.l2(x))
        x = nn.ReLU(self.l3(x))
        x = nn.ReLU(self.l4(x))
        x = self.lout(x)
        return x