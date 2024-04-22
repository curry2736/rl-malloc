from request_streams.allocator_bad_base import AllocatorBadBase
import random



class WFGood(AllocatorBadBase):
    def __init__(self, page_size=100, *args, **kwargs):
        super().__init__(page_size, *args, **kwargs)
    def _create_trajectory(self):
        bigger_alloc_ratios = [8,9,10]
        bigger_alloc_sizes = [self.page_size//rt for rt in bigger_alloc_ratios]
        small_alloc_sizes = [1,2,3]
        large_section_size = self.page_size//2.5

        initial_free_amt = self.page_size 
        trajs = []
        num_allocs = 0
        while True:
            alloc_size = random.choice(bigger_alloc_sizes)
            if initial_free_amt - alloc_size <= large_section_size:
                break
            num_allocs +=1
            initial_free_amt -=alloc_size
            trajs.append((1,alloc_size, 0))
        
        trajs.append((1,  int(initial_free_amt - large_section_size) , 0))
        num_frees = random.choice([1,2,3])
        assert num_frees < num_allocs
        for i in range(num_allocs):
            trajs.append(("free", 0, 0))
            if i < num_frees:
                trajs.append((1, random.choice(small_alloc_sizes), 0))
        
        initial_alloc_amt = self.page_size - initial_free_amt
        
        trajs.append((1, initial_alloc_amt, 0))




        return trajs
    