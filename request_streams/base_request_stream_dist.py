from request_streams.base_request_stream import BaseRequestStream
import numpy as np
from typing import List

class BaseRequestStreamDist(BaseRequestStream):
    def __init__(self, name = "BaseRequestStreamDist", alloc_sizes: List[int]=[1,2,3,4,5], alloc_probs: List[float] = [0.2,0.2,0.2,0.2,0.2], free_prob: float=0.4):
        """
        Initializes a BaseRequestStreamTraj object.

        Args:
            alloc_sizes (list[int]): A list of allocation sizes.
            alloc_probs (list[float]): A list of probabilities corresponding to the allocation sizes.
            free_prob (float): The probability of a free request.
        """
        
        assert(sum(alloc_probs) == 1)

        super().__init__(name)
        self.alloc_sizes = alloc_sizes
        self.alloc_probs = alloc_probs
        self.free_prob = free_prob
        self.allocated_indices = []

    def add_to_allocated_indices(self, index):
        self.allocated_indices.append(index)
    
    def remove_from_allocated_indices(self, index):
        self.allocated_indices.remove(index)

    def get_next_req(self):
        """
        return the next allocation request

        output:
            tuple(free_or_alloc: {0,1}, mem_addr_or_amt: int, new_traj: {0, 1})
        """
        #print("In get_next_req()")
        new_traj = False
        free_or_alloc = np.random.choice(np.array([0,1]), p=[self.free_prob, 1-self.free_prob])
        #print("    free_or_alloc: ", free_or_alloc, "allocated_indices: ", self.allocated_indices)
        if len(self.allocated_indices) <= 0:
            free_or_alloc = 1
        if free_or_alloc == 0:
            mem_addr_or_amt = np.random.choice(self.allocated_indices)
        else:
            mem_addr_or_amt = np.random.choice(self.alloc_sizes, p=self.alloc_probs)
        return (free_or_alloc, mem_addr_or_amt, int(new_traj))
