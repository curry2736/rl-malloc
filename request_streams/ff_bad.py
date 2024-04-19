from request_streams.allocator_bad_base import AllocatorBadBase
import random

class FFBad(AllocatorBadBase):
    def _create_trajectory(self):
        big_ratio = random.choice([2,3,4,5])
        small_ratios = [10, 20,30,50,100, 200]
        small_ratios = [sr for sr in small_ratios if sr < self.page_size ]

        self.big_alloc_size = self.page_size//big_ratio
        small_alloc_sizes = [self.page_size//sr for sr in small_ratios ]


        trajectories = []
        # initial_allocs = random.choice([1,2,3])
        # trajectories = [(1, random.choice(small_alloc_sizes), 0)] * initial_allocs
        alloc = (1, self.big_alloc_size, 0)
        curr_allocated = self.big_alloc_size
        print("sizes ",self.page_size, self.big_alloc_size)
        while curr_allocated < self.page_size - self.big_alloc_size:
            trajectories.append(alloc)
            alloc = (1, random.choice(small_alloc_sizes), 0)
            curr_allocated += alloc[1]
        
        # trajectories.extend([(0, 0, 0), (1,1,0), (1, big_alloc_size, 0)])
        self.traj_len = len(trajectories) + 3
        self.initial_traj_len = len(trajectories)
        return trajectories
    
    def get_next_req(self):
        if self.ptr < self.traj_len:
            print(self.ptr, self.allocated_indices, self.initial_traj_len)
            if self.ptr >= self.initial_traj_len:
                #free big (first) allocation, do a minimal allocation, then allocate big again
                
                self.traj.extend([(0, self.allocated_indices[0], 0), (1,1,0), (1, self.big_alloc_size, 0)])

            ret = self.traj[self.ptr]
            self.ptr+=1
            return ret
        return super().get_next_req()
