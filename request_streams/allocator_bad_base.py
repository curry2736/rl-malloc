# s s Biiiiigggggg (50) .... allocate randomly until the free space is less than big, but its more than small valye like 2
# free the big ss______________________sssbs_, allocate 2

#ss Biiiiig sssbss___________ allocate the big number agai
from request_streams.base_request_stream_dist import BaseRequestStreamDist
import numpy as np
from typing import List, Tuple

class AllocatorBadBase(BaseRequestStreamDist):
    def __init__(self, page_size=100):
        super().__init__("First fit bad stream")
        self.page_size = page_size
        self.ptr = 0
        self.traj = self._create_trajectory()

    
    def _create_trajectory(self):
        raise NotImplementedError
    
    