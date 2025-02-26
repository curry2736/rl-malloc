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
    

    def action(self, state: dict, history=None):
        page = state["pages"] 
        rq = state["rq"]
        if rq[0] != 1:
            print(rq[0])
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
                #assert allocated_index not in page_copy[0].allocated_list
                res = page_copy[0].allocate(allocated_index, alloc_size) #TODO: Remember this is only for single page env
                assert res
                #rint(page_copy[0])
                afterstate_values.append(self.value_network((page_copy[0], history)))
                possible_actions.append(allocated_index)
            #check if afterstate_values has more than 1 unique value
            #print("possible_actions: ", possible_actions)
            # if len(set(afterstate_values)) > 1:
            #     print("afterstate_values: ", afterstate_values)

        #apply softmax on afterstate_values
        temperature = .1
        afterstate_values = np.array(afterstate_values).reshape(-1,)
        #print(afterstate_values)
        afterstate_values = torch.tensor(afterstate_values, dtype=torch.float32)
        afterstate_values = torch.nn.functional.softmax(afterstate_values / temperature, dim=0)

        #pick from distribution
        best_action_index = np.random.choice([0, 1, 2], p=afterstate_values.numpy())
        if best_action_index == 2:
            #print("best_action_index: ", best_action_index)
            pass
        return (1, possible_actions[best_action_index], best_action_index)
