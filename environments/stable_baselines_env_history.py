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

class StableBaselineEnvHistory(BaseEnv):
    def __init__(self, history_len=0, page_size=50, allocator="ff_bad"):
        super().__init__(page_size=page_size, allocator=allocator)
        self.history_len = history_len
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=page_size, shape=(1, page_size + 1 + history_len), dtype=np.int32)
        self.alloc_history = []
        self.history = []
        for i in range(history_len):
            self.history.append(-1) #alloc size
        self.history = np.array(self.history, dtype=np.int32)
        self.history_idx = self.history.shape[0]-1

    def _get_state(self, rq):
        bm_copy = self.page.bitmap.copy()
        rq = np.array([rq[1]], dtype=np.int32)
        #print("rq: ", rq)
        history = self.history.copy()
        #print(self.history, self.history_idx)
        #print(bm_copy.shape, rq.shape, history.shape)
        return np.concatenate([bm_copy, rq, history], axis=0)[None, :]

    def reset(self, seed=None, options=None):
        obs,_ = super().reset()
        self.history = []
        for i in range(self.history_len):
            self.history.append(-1) #alloc size
        self.history = np.array(self.history, dtype=np.int32)
        self.history_idx = self.history.shape[0]-1
        return obs, {}

    #action is 0,1,2 corresponding to ff, bf, and wf,  
    def step(self, action):
        allocators = [BestFitAllocator(), WorstFitAllocator(), FirstFitAllocator()]
        allocator = allocators[action]
        
        while self.prev_request[0] == 0 or self.prev_request[0] == -1:
            state, _, done = super().step([self.prev_request[0], self.prev_request[1]])

        _, allocated_index = allocator.handle_alloc_req([self.page], self.prev_request[1])

        if self.history_len > 0:
            self.history[self.history_idx] = self.prev_request[1]
        self._update_history_idx()

        state, reward, done = super().step([1,allocated_index])

        while self.prev_request[0] == 0 or self.prev_request[0] == -1:
            state, _, done = super().step([self.prev_request[0], self.prev_request[1]])

        return state, reward, done, False, {}
    
    def render(self):
        print(self.page.bitmap)
    


