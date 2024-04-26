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
from environments.stable_baselines_env_history import StableBaselineEnvHistory

import numpy as np

class StableBaselineEnvLinearHistory(BaseEnv):
    def __init__(self, history_len=0, page_size=50, allocator="ff_bad"):
        super().__init__(page_size=page_size, allocator=allocator)
        
        self.feature_len = 9
        self.history_len = history_len
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=page_size+1, shape=(1, self.feature_len + 1 + self.history_len), dtype=np.float32)

        self.history = []
        for i in range(history_len):
            self.history.append(-1) #alloc size
        self.history = np.array(self.history, dtype=np.int32)
        self.history_idx = self.history.shape[0]-1

    def _get_state(self, rq):
        bm_copy = self.page.bitmap.copy()
        if len(self.page.free_list) > 0:
            avg_free_idx = np.mean([x["idx"] for x in self.page.free_list])
            avg_free_block_size = np.mean([x["size"] for x in self.page.free_list])
            largest_free_block = max([x["size"] for x in self.page.free_list])
            smallest_free_block = min([x["size"] for x in self.page.free_list])
        else:
            avg_free_idx = self.page_size
            avg_free_block_size = self.page_size
            largest_free_block = 0
            smallest_free_block = 0
        
        if len(self.page.allocated_list) > 0:
            avg_alloc_block_size =  np.mean([v for v in self.page.allocated_list.values()])
            avg_alloc_idx = np.mean([k for k in self.page.allocated_list.keys()])
            total_allocated_block_size = np.sum([v for v in self.page.allocated_list.values()])
            smallest_alloc_block = min([v for v in self.page.allocated_list.values()])
            largest_alloc_block = max([v for v in self.page.allocated_list.values()])
        else:
            avg_alloc_block_size = self.page_size
            avg_alloc_idx = self.page_size
            total_allocated_block_size = 0
            smallest_alloc_block = 0
            largest_alloc_block = 0

        features = np.array([avg_free_idx, avg_free_block_size, avg_alloc_idx, avg_alloc_block_size, largest_free_block, smallest_free_block, total_allocated_block_size, smallest_alloc_block, largest_alloc_block], dtype=np.float32)
        rq = np.array([rq[1]], dtype=np.float32)

        return np.concatenate([features, rq, self.history], axis=0)[None, :]
    
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

        _, allocated_index = allocator.handle_alloc_req([self.page], self.prev_request[1])

        if self.history_len > 0:
            self.history[self.history_idx] = self.prev_request[1]
        self._update_history_idx()

        state, reward, done = super().step([1,allocated_index])

        while self.prev_request[0] == 0:
            state, _, done = super().step([self.prev_request[0], self.prev_request[1]])
        #print(state, state.shape)
        return state, reward, done, False, {}
    
    def render(self):
        print(self.page.bitmap)
    


    