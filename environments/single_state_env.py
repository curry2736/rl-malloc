from base_env import BaseEnv
import numpy as np

class SingleStateEnv(BaseEnv):
    def __init__(self, allocator="trajectory", invalid_action_reward=0, done_reward=-1000) -> None:
        super().__init__(allocator, invalid_action_reward, done_reward)
    
    def _get_state(self, rq):
        return np.concatenate(self.page.bitmap, np.array([rq[0], rq[1]]))
    