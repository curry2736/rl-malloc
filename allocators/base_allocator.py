from page import Page

class BaseAllocator:
    def __init__(self):
        self.pages_in_use = [] #list[Page]
    
    def handle_alloc_req(self, alloc_size:int):
        """
        handle the allocation request
        
        input:
            alloc_size: size of the requested memory
        output:
            i, j: page index and block index within page
        """
        raise NotImplementedError()


