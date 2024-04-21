import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class OneLayerNN(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            #nn.Conv1d(in_dims, 32, 1),
            nn.Linear(in_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.optimizer = torch.optim.Adam(self.linear_relu_stack.parameters(), lr=0.001, betas=(0.9,0.999))
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        res = self.linear_relu_stack(x)
        #print(x, res)
        return res

class LinearValueFn():
    def __init__(self, history_len=0):
        self.num_features = 9 #change if adding more features
        self.weights = np.random.rand(self.num_features)
        self.nn = OneLayerNN(self.num_features)
        #print("self.weights initialized to ", self.weights)

    def __call__(self,s):
        self.nn.eval()
        return self.nn(self.extract_features(s)).item()
        #return np.dot(self.extract_features(s), self.weights)

    def extract_features(self, s):
        num_free_blocks = len(s.free_list)
        num_allocated_blocks = len(s.allocated_list)
        page_size = s.page_size
        avg_free_block_size = np.mean([x["size"] for x in s.free_list])
        if len(s.allocated_list) > 0:
            avg_allocated_block_size =  np.mean([v for v in s.allocated_list.values()])
            avg_allocated_list_idx = np.mean([k for k in s.allocated_list.keys()])
            total_allocated_block_size = np.sum([v for v in s.allocated_list.values()])
        else:
            avg_allocated_block_size = 0
            avg_allocated_list_idx = -1
            total_allocated_block_size = 0
        total_free_block_size = np.sum([x["size"] for x in s.free_list])
        avg_free_list_idx = np.mean([x["idx"] for x in s.free_list])
        
        # #normalize features to be between 0 and 1
        # num_free_blocks = (num_free_blocks - 0) / (s.page_size - 0)
        # num_allocated_blocks = (num_allocated_blocks - 0) / (s.page_size - 0)
        # avg_free_block_size = (avg_free_block_size - 0) / (s.page_size - 0)
        # avg_allocated_block_size = (avg_allocated_block_size - 0) / (s.page_size - 0)
        # total_free_block_size = (total_free_block_size - 0) / (s.page_size - 0)
        # total_allocated_block_size = (total_allocated_block_size - 0) / (s.page_size - 0)


        #TODO: history of requests

        return np.array([num_free_blocks, num_allocated_blocks, page_size, avg_free_block_size, avg_allocated_block_size, total_free_block_size, total_allocated_block_size, avg_free_list_idx, avg_allocated_list_idx])
        

    def update(self, alpha, G, s_tau):
        extracted_features = self.extract_features(s_tau)
        self.nn.train()
        self.nn.optimizer.zero_grad()
        value = self.nn.forward(extracted_features)
        loss = self.nn.loss_function(value, torch.tensor(G, dtype=torch.float32))
        #print(value, G, loss.item())
        loss.backward()
        self.nn.optimizer.step()
        return loss.item()

        # #print("extracted_features: ", extracted_features)
        # print(G, self(s_tau))
        # self.weights = self.weights + alpha * (G - self(s_tau)) * extracted_features
        # print("self.weights changed to ", self.weights)
        # return abs(G - self(s_tau))
        

