from request_streams.allocator_bad_base import AllocatorBadBase
import random

class FFBad(AllocatorBadBase):
    def _create_trajectory(self):
        big_ratio = random.choice([2,3,4,5])
        small_ratios = [10, 20,30,50,100, 200]
        small_ratios = [sr for sr in small_ratios if sr < self.page_size]

        self.big_alloc_size = self.page_size//big_ratio
        small_alloc_sizes = [self.page_size//sr for sr in small_ratios]

        trajectories = []
        alloc = (1, self.big_alloc_size, 0)
        curr_allocated = self.big_alloc_size
        trajectories.append(alloc)
        
        while True:
            current_free_space = self.page_size - curr_allocated
            alloc = (1, random.choice(small_alloc_sizes), 0)
            if current_free_space - alloc[1] < self.big_alloc_size:
                break
            curr_allocated += alloc[1]
            trajectories.append(alloc)

        #print(self.page_size, curr_allocated, self.big_alloc_size) #it is possible the amount of free space is already less than the big alloc
        #allocate enough to where the amount of free space is barely less than the big alloc
        free_space = self.page_size - curr_allocated
        final_alloc = (1, free_space - self.big_alloc_size +1, 0)
        trajectories.append(final_alloc)
        trajectories.extend([("free", 0, 0), (1,1,0), (1, self.big_alloc_size, 0)])
        self.traj_len = len(trajectories) #+ 3
        self.initial_traj_len = len(trajectories)
        return trajectories
    
    
