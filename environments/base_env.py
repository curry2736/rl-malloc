from gym import Env, spaces
from ..page import Page
from ..request_streams.base_request_stream_traj import BaseRequestStreamTraj
import numpy as np

class BaseEnv():
    def __init__(self, allocator="trajectory", invalid_action_reward=0, done_reward = -1000) -> None:
        if allocator == "trajectory":
            self.request_stream_cls = BaseRequestStreamTraj
        else:
            raise NotImplementedError()
        
        #do not assign them until reset is called!
        self.invalid_action_reward = invalid_action_reward
        self.done_reward = done_reward
        self.page = None
        self.request_stream = None
        self.prev_request = None

    def _get_state(self, rq):
        NotImplementedError()

    def reset(self):
        self.request_stream = self.request_stream_cls()
        self.page = Page()
        first_rq = self.request_stream.get_next_req()
        self.prev_request = first_rq
        return self._get_state(first_rq), False
        

    def step(self, action):
        allocation_success = self.page.allocate(action)
        if not allocation_success:
            return self._get_state(self.prev_request), self.invalid_action_reward, False
        next_rq = self.request_stream.get_next_req()
        state = self._get_state(next_rq)
        done = next_rq[2] or ( next_rq[0] and not self.page.space_available(next_rq[1]))
        reward = self.done_reward if done else 1
        self.prev_request = next_rq
        return state, reward, done

    def close(self):
        pass



