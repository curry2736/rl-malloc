import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
class NeuralNetwork(nn.Module):
    def __init__(self, in_dims):
        super().__init__()


        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.optimizer = torch.optim.Adam(self.linear_relu_stack.parameters(), lr=0.001, betas=(0.9,0.999))
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        res = self.linear_relu_stack(x)
        #print(res + 5)
        return res  #optimistic initialization

class NNValueFn():
    def __init__(self, state_dims, history_len=0):
        """
        state_dims: the number of dimensions of state space
        """
        self.history_len = 2 * history_len
        self.model = NeuralNetwork(state_dims + self.history_len)

    def __call__(self,s):
        
        history = np.array(s[1]).reshape(-1,)
        s = s[0]
        s = s.bitmap
        #s = np.concatenate((s, history))
        self.model.eval()
        return (self.model(torch.tensor(s, dtype=torch.float32)).detach().numpy())[0]

    def update(self, alpha, G, s_tau):
        #print(s_tau)
        history = np.array(s_tau[1]).reshape(-1,)
        s_tau = s_tau[0] #s_tau is a page object
        s_tau = s_tau.bitmap 
        #s_tau = np.concatenate((s_tau, history))
        self.model.train()
        self.model.optimizer.zero_grad()
        loss = self.model.loss_function(self.model(torch.tensor(s_tau, dtype=torch.float32))[0], torch.tensor(G, dtype=torch.float32))
        #print(self.model(torch.tensor(s_tau, dtype=torch.float32))[0], torch.tensor(G, dtype=torch.float32), loss.item())
        loss.backward()
        self.model.optimizer.step()

        return loss.item()

