from page import Page
import copy
from allocators.best_fit_allocator import BestFitAllocator
from allocators.worst_fit_allocator import WorstFitAllocator
from allocators.first_fit_allocator import FirstFitAllocator
import numpy as np
import torch



class NNFitPolicy():
    def __init__(self, value_network) -> None:
        self.value_network = value_network
    

    def action(self, state: dict):
        page = state["pages"] 
        rq = state["rq"]
        assert rq[0] == 1
        alloc_size = rq[1]

        allocators = [BestFitAllocator(), WorstFitAllocator(), FirstFitAllocator()]
        afterstate_values = []
        possible_actions = []

        #0 -> first fit, 1 -> best fit, 2 -> worst fit
        with torch.no_grad():
            for allocator in allocators:
                page_copy = copy.deepcopy(page)
                allocated_page, allocated_index = allocator.handle_alloc_req(page_copy, alloc_size)
                res = page_copy[0].allocate(allocated_index, alloc_size) #TODO: Remember this is only for single page env
                assert res
                #rint(page_copy[0])
                afterstate_values.append(self.value_network(page_copy[0]))
                possible_actions.append(allocated_index)
            #print(afterstate_values)

        
        best_action_index = np.argmax(afterstate_values)
        return (1, possible_actions[best_action_index], best_action_index)
