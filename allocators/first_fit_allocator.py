from allocators.base_allocator import BaseAllocator

class FirstFitAllocator(BaseAllocator):
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
        #print("In handle_alloc_req(), Handling allocation request of size: ", alloc_size)
        for i, page in enumerate(pages):
            #print("    Current page's free list: ", page.free_list, "Current page's allocated list: ", page.allocated_list)
            for j, block in enumerate(page.free_list):
                if block["size"] >= alloc_size:
                    return i, block["idx"]
        print("ASDJKLAKJSLDJKLDASJKL")
        return (-1, -1)