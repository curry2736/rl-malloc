import torch
import torch.nn as nn
from tqdm import tqdm

class ConvNN(nn.Module):
    def __init__(self, in_dims):
        super().__init__()


        self.linear_relu_stack = nn.Sequential(
            nn.Conv1d(1, 32, 1),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.optimizer = torch.optim.Adam(self.linear_relu_stack.parameters(), lr=0.001, betas=(0.9,0.999))
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x[None, :]
        print("x.shape", x.shape)
        res = self.linear_relu_stack(x)
        #print("res.shape", res.shape)
        return res

class ConvNNValueFn():
    def __init__(self, state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        self.model = ConvNN(state_dims)

    def __call__(self,s):
        self.model.eval()
        s = s.bitmap
        print(s.shape)
        return (self.model(torch.tensor(s, dtype=torch.float32)).detach().numpy())[0]

    def update(self, alpha, G, s_tau):
        #print(s_tau)
        s_tau = s_tau.bitmap #s_tau is a page object
        #print("s_tau.shape", s_tau.shape)

        self.model.train()
        self.model.optimizer.zero_grad()
        loss = self.model.loss_function(self.model(torch.tensor(s_tau, dtype=torch.float32))[0], torch.tensor(G, dtype=torch.float32))
        #print(self.model(torch.tensor(s_tau[:, None], dtype=torch.float32))[0], torch.tensor(G, dtype=torch.float32), loss.item())
        loss.backward()
        self.model.optimizer.step()

        return loss.item()

