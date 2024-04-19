import numpy as np

class LinearValueFn():
    def __init__(self, history_len=0):
        self.num_features = 7 #change if adding more features
        self.weights = np.random.rand(self.num_features)

    def __call__(self,s):
        return np.dot(self.extract_features(s), self.weights)

    def extract_features(self, s):
        num_free_blocks = len(s.free_list)
        num_allocated_blocks = len(s.allocated_list)
        page_size = s.page_size
        avg_free_block_size = np.mean([x["size"] for x in s.free_list])
        avg_allocated_block_size = np.mean([v for v in s.allocated_list.values()])
        total_free_block_size = np.sum([x["size"] for x in s.free_list])
        total_allocated_block_size = np.sum([v for v in s.allocated_list.values()])
        #TODO: history of requests

        return np.array([num_free_blocks, num_allocated_blocks, page_size, avg_free_block_size, avg_allocated_block_size, total_free_block_size, total_allocated_block_size])
        

    def update(self, alpha, G, s_tau):
        self.weights = self.weights + alpha * (G - self(s_tau)) * self.extract_features(s_tau)
        

