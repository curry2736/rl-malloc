import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
class ConvNN(nn.Module):
    def __init__(self, in_dims):
        super().__init__()


        self.linear_relu_stack = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.MaxPool1d(4),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=0),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.optimizer = torch.optim.Adam(self.linear_relu_stack.parameters(), lr=0.001, betas=(0.9,0.999))
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x[None, :]
        #print("x.shape", x.shape)
        res = self.linear_relu_stack(x)
        #print("res.shape", res.shape)
        return res #optimistic initialization

class ConvNNValueFn():
    def __init__(self, state_dims, use_history=True):
        """
        state_dims: the number of dimensions of state space
        """
        self.model = ConvNN(state_dims)
        self.use_history = use_history

    def __call__(self,s):
        self.model.eval()
        history = np.array(s[1]).reshape(-1,)
        s = s[0]
        s = s.bitmap
        if self.use_history:
            s = np.concatenate((s, history))
        return (self.model(torch.tensor(s, dtype=torch.float32)).detach().numpy())[0]

    def update(self, alpha, G, s_tau):
        #print(s_tau)
        history = np.array(s_tau[1]).reshape(-1,)
        s_tau = s_tau[0]
        s_tau = s_tau.bitmap #s_tau is a page object
        s_tau = np.concatenate((s_tau, history))
        #print("s_tau.shape", s_tau.shape)

        self.model.train()
        self.model.optimizer.zero_grad()
        #print(self.model(torch.tensor(s_tau, dtype=torch.float32)).shape, torch.tensor(G, dtype=torch.float32).squeeze().shape)
        loss = self.model.loss_function(self.model(torch.tensor(s_tau, dtype=torch.float32)).squeeze(), torch.tensor(G, dtype=torch.float32).squeeze())
        #print(self.model(torch.tensor(s_tau[:, None], dtype=torch.float32))[0], torch.tensor(G, dtype=torch.float32), loss.item())
        loss.backward()
        self.model.optimizer.step()

        return loss.item()

