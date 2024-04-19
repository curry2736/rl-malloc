from environments.base_env import BaseEnv
import numpy as np

class SingleStateEnv(BaseEnv):
    def __init__(self, allocator="dist", invalid_action_reward=0, done_reward=0, trajs=None, page_size=256) -> None:
        super().__init__(allocator, invalid_action_reward, done_reward, trajs=trajs, page_size=page_size)
    
    def _get_state(self, rq):
        #print("in _get_state()")
        #print(f"    Next state request contents - free_or_alloc: {rq[0]}, mem_addr_or_amt: {rq[1]}, int(new_traj): {rq[2]})")
        st =  {
                "rq": np.array([rq[0], #free or alloc
                        rq[1],  #mem_addr_or_amt
                        rq[2], #new_traj bool
                        ]),
                "pages": [self.page]}
        return st
    