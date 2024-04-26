import gymnasium as gym
from page import Page
from request_streams.base_request_stream_traj import BaseRequestStreamTraj
from request_streams.base_request_stream_dist import BaseRequestStreamDist
from request_streams.ff_bad import FFBad
from request_streams.wf_good import WFGood
from request_streams.mixed_ff_bad_wf_good import MixedFFBadWFGood


import numpy as np

class BaseEnv(gym.Env):
    def __init__(self, allocator="dist", invalid_action_reward=-10, done_reward = 0, trajs=None, page_size=256, allocator_kwargs = None) -> None:
        
        #do not assign them until reset is called!
        self.invalid_action_reward = invalid_action_reward
        self.done_reward = done_reward
        self.page = None
        self.request_stream = None
        self.prev_request = None
        self.trajs = trajs
        self.page_size = page_size
        self.horizon = page_size * 2
        self.ts = 0

        self.allocator_kwargs = allocator_kwargs if allocator_kwargs is not None else {}
        if allocator == "trajectory":
            self.request_stream_cls = BaseRequestStreamTraj
            self.allocator_kwargs["trajectories"] = self.trajs
        elif allocator == "dist":
            self.request_stream_cls = BaseRequestStreamDist
        elif allocator == "ff_bad":
            self.request_stream_cls = FFBad
            self.allocator_kwargs["page_size"] = self.page_size
        elif allocator == "wf_good":
            self.request_stream_cls = WFGood
            self.allocator_kwargs["page_size"] = self.page_size
        elif allocator == "mixed_ff_bad_wf_good":
            self.request_stream_cls = MixedFFBadWFGood
            self.allocator_kwargs["page_size"] = self.page_size
        else:
            raise NotImplementedError()

        print(self.request_stream_cls)


    def _get_state(self, rq):
        NotImplementedError()

    def reset(self, seed=None):
        self.request_stream = self.request_stream_cls(**self.allocator_kwargs)
        #print("self.request_stream: ", self.request_stream)
        self.page = Page(page_size=self.page_size)
        first_rq = self.request_stream.get_next_req()
        self.prev_request = first_rq
        self.freed_list = []
        self.has_freed = False
        self.ts = 0
        return self._get_state(first_rq), False
        

    def step(self, action): #action = (free_or_alloc, mem_addr_or_amt)
        #print(action)
        allocate = action[0]
        #print("allocate: ", allocate)
        # print("allocated indices rq stream: ", self.request_stream.allocated_indices)
        # print("allocated indices page: ", self.page.allocated_list)
        if allocate == 1:
            self.ts +=1

            assert not isinstance(action[1], np.ndarray)
            allocation_success = self.page.allocate(action[1], self.prev_request[1])
            if not allocation_success:
                # print("failed alloc")
                return self._get_state(self.prev_request), self.invalid_action_reward, False
            else:
                self.request_stream.add_to_allocated_indices(action[1])
            #print("allocated at ", action[1])
        elif allocate == -1: #free the first thing allocation list
            allocated_addresses = list(self.page.allocated_list.keys())
            if len(allocated_addresses) != 0:
                # print("allocated_addresses: ", allocated_addresses)
                # print("freeing at ", allocated_addresses[0])

                self.freed_list.append(allocated_addresses[0])
                self.page.free(allocated_addresses[0])
                self.request_stream.remove_from_allocated_indices(allocated_addresses[0])
        else:
            self.freed_list.append(action[1])
            #print("freed indicies", self.freed_list)
            self.page.free(action[1])
            self.request_stream.remove_from_allocated_indices(action[1])

        next_rq = self.request_stream.get_next_req()
        state = self._get_state(next_rq)
        done = self.ts > self.horizon or next_rq[2] or ( next_rq[0] and not self.page.space_available(next_rq[1])) 
        
        reward = self.done_reward if (done) else .1 #if (done or not self.has_freed) else 1

        # if self.ts > self.horizon:
        #     print(self.ts)
        #     assert done, f"done should be true if super {self.horizon} < {self.ts}"
        self.prev_request = next_rq
        #print(reward, self.has_freed)
        if allocate == 0:
            self.has_freed = True

        return state, reward, done

    def close(self):
        pass

    def _update_history_idx(self):
        self.history_idx -= 1
        if self.history_idx < 0:
            self.history_idx = self.history.shape[0]-1



