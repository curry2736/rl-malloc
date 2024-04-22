from request_streams.base_request_stream_dist import BaseRequestStreamDist
import numpy as np
from typing import List, Tuple

class AllocatorBadBase(BaseRequestStreamDist):
    def __init__(self, page_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_size = page_size
        self.ptr = 0
        self.traj = self._create_trajectory()

    
    def _create_trajectory(self):
        raise NotImplementedError
    
    def get_next_req(self):
        if self.ptr < len(self.traj):
            if self.traj[self.ptr][0] == "free":
                free_ind = self.traj[self.ptr][1]
                ret = (0, self.allocated_indices[free_ind], 0)
            else:
                ret = self.traj[self.ptr]
            self.ptr+=1
            return ret
        #print("Done with trajectory")
        return super().get_next_req()
    
    