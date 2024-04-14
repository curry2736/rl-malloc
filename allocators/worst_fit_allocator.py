from allocators.base_allocator import BaseAllocator

class WorstFitAllocator(BaseAllocator): #pick the free block with the largest amount of space available
    def __init__(self):
        super().__init__()
    
    def handle_alloc_req(self, alloc_size:int):
        """
        handle the allocation request
        
        input:
            alloc_size: size of the requested memory
        output:
            i, j: page index and block index within page
        """
        worst_fit_size = -1
        worst_fit_idxs = (-1, -1)
        for i, page in enumerate(self.pages_in_use):
            free_list = page.free_list
            free_list_sorted = sorted(free_list, key=lambda x: x["size"], reverse=True)

            worst_block = free_list_sorted[0]

            if worst_block["size"] >= alloc_size and worst_block["size"] > worst_fit_size:
                worst_fit_size = worst_block["size"]
                worst_fit_idxs = (i, worst_block["idx"])
                break

        return worst_fit_idxs[0], worst_fit_idxs[1]