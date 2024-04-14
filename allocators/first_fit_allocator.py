from allocators.base_allocator import BaseAllocator

class FirstFitAllocator(BaseAllocator):
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
        for i, page in enumerate(self.pages_in_use):
            print("Current page's free list: ", page.free_list)
            for j, block in enumerate(page.free_list):
                if block["size"] >= alloc_size:
                    return i, block["idx"]
        return -1, -1