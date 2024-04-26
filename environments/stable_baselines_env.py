import gymnasium as gym
from page import Page
from request_streams.base_request_stream_traj import BaseRequestStreamTraj
from request_streams.base_request_stream_dist import BaseRequestStreamDist
from request_streams.ff_bad import FFBad
from request_streams.wf_good import WFGood
from environments.base_env import BaseEnv
from allocators.best_fit_allocator import BestFitAllocator
from allocators.worst_fit_allocator import WorstFitAllocator
from allocators.first_fit_allocator import FirstFitAllocator
import copy

import numpy as np

class StableBaselineEnv(BaseEnv):
    def __init__(self, history_len=0, page_size=50, allocator="ff_bad",high_level_actions=True):
        super().__init__(page_size=page_size, allocator=allocator)
        self.high_level_actions = high_level_actions
        self.history_len = history_len
        if high_level_actions:
            self.action_space = gym.spaces.Discrete(3)
        else:
            self.action_space = gym.spaces.Discrete(page_size)
        self.observation_space = gym.spaces.Box(low=0, high=page_size, shape=(1, page_size + 1), dtype=np.uint8)

    def _get_state(self, rq):
        bm_copy = self.page.bitmap.copy()
        rq = np.array([rq[1]], dtype=np.uint8)
        # print(bm_copy, np.array([rq[1]]))
        # print(type(bm_copy), type(np.array([rq[1]])))
        # print(bm_copy.shape, np.array([rq[1]]).shape)
        # print(bm_copy.dtype, np.array([rq[1]]).dtype)
        return np.concatenate([bm_copy, rq], axis=0)[None, :]
    
    def reset(self, seed=None, options=None):
        obs,_ = super().reset()
        return obs, {}
    
    #action is 0,1,2 corresponding to ff, bf, and wf,  
    def step(self, action):
        if self.high_level_actions:
            allocators = [BestFitAllocator(), WorstFitAllocator(), FirstFitAllocator()]
            allocator = allocators[action]

            _, allocated_index = allocator.handle_alloc_req([self.page], self.prev_request[1])
        else:
            allocated_index = int(action)
        #assert allocated_index not in page_copy[0].allocated_list
        #rint(page_copy[0])
        state, reward, done = super().step([1,allocated_index])

        while self.prev_request[0] == 0:
            state, _, done = super().step([self.prev_request[0], self.prev_request[1]])
        
        truncated = self.ts > self.horizon

        return state, reward, done, False, {}
    
    def render(self):
        print(self.page.bitmap)
    



