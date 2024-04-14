from request_streams.base_request_stream import BaseRequestStream
import numpy as np


class BaseRequestStreamDist(BaseRequestStream):
    def __init__(self, alloc_sizes: list[int], alloc_probs: list[float], free_prob: list[float]):
        """
        Initializes a BaseRequestStreamTraj object.

        Args:
            alloc_sizes (list[int]): A list of allocation sizes.
            alloc_probs (list[float]): A list of probabilities corresponding to the allocation sizes.
            free_prob (float): The probability of a free request.
        """
        
        assert(sum(alloc_probs) == 1)

        super().__init__("BaseRequestStreamDist")
        self.alloc_sizes = alloc_sizes
        self.alloc_probs = alloc_probs
        self.free_prob = free_prob
        self.allocated_indices = []

    def get_next_req(self):
        """
        return the next allocation request

        output:
            tuple(free_or_alloc: {0,1}, mem_addr_or_amt: int, new_traj: {0, 1})
        """
        new_traj = False
        free_or_alloc = np.random.choice(np.array([0,1]), p=[self.free_prob, 1-self.free_prob])
        if free_or_alloc == 0 and self.allocated_indices:
            mem_addr_or_amt = np.random.choice(self.allocated_indices)
            self.allocated_indices.remove(mem_addr_or_amt)
        else:
            mem_addr_or_amt = np.random.choice(self.alloc_sizes, p=self.alloc_probs)
            self.allocated_indices.append(mem_addr_or_amt)
        return (free_or_alloc, mem_addr_or_amt, int(new_traj))
