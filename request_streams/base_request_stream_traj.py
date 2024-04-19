from request_streams.base_request_stream import BaseRequestStream
import numpy as np
from typing import List, Tuple

class BaseRequestStreamTraj(BaseRequestStream):
    def __init__(self, trajectories: List[List[Tuple[int, int]]], ): #create from a list of trajectories
        """
        Initializes a BaseRequestStreamTraj object.

        Args:
            trajectories (list[list[tuple[int, int]]]): A list of trajectories, where each trajectory is a list of tuples
                representing the request type (free or alloc) and size of allocation / address to free.

                0 is free, 1 is alloc
        Returns:
            None
        """
        super().__init__("BaseRequestStreamTraj")
        self.trajectories = trajectories
        self.curr_trajectory = None
        self.ptr_in_trajectory = 0

    def create_trajectories(self, bad_for="first", length=1000, alloc_sizes=[1,2,3,4,50], free_prob=0.2, save=False):
        assert bad_for in ["best", "first", "worst"]
        traj = []
        allocated_indices = set()
        while len(traj) < length:
            break
            
            


    def get_next_req(self):
        """
        return the next allocation request

        output:
            tuple(free_or_alloc: {0,1}, mem_addr_or_amt: int, new_traj: bool)
        """
        new_traj = False
        if self.curr_trajectory is None or self.ptr_in_trajectory >= len(self.curr_trajectory):
            new_trajectory_idx = self._sample_request_trajectory()
            self.curr_trajectory = self.trajectories[new_trajectory_idx]
            self.ptr_in_trajectory = 0
            new_traj = True
        
        free_or_alloc, mem_addr_or_amt = self.curr_trajectory[self.ptr_in_trajectory]
        self.ptr_in_trajectory += 1
        return (free_or_alloc, mem_addr_or_amt, int(new_traj))
    
    def _sample_request_trajectory(self):
        """
        sample a trajectory of allocation requests
        """
        
        #sample from uniform distribution ranging from 0 to len(trajectories)
        trajectory_idx = np.random.randint(0, len(self.trajectories))
        return trajectory_idx
