from allocators.base_allocator import BaseAllocator

class BestFitAllocator(BaseAllocator): #pick the free block with the least amount of space available as long as the alloc req can fit
    def __init__(self):
        super().__init__()
    
    def handle_alloc_req(self, pages: list, alloc_size:int):
        """
        handle the allocation request
        
        input:
            alloc_size: size of the requested memory
        output:
            i, j: page index and block index within page
        """
        best_fit_size = float('inf')
        best_fit_idxs = (-1, -1)
        for i, page in enumerate(pages):
            free_list = page.free_list
            free_list_sorted = sorted(free_list, key=lambda x: x["size"])
            for block in free_list_sorted:
                if block["size"] >= alloc_size and block["size"] < best_fit_size:
                    best_fit_size = block["size"]
                    best_fit_idxs = (i, block["idx"])
                    break

        return best_fit_idxs[0], best_fit_idxs[1]