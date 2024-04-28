from request_streams.allocator_bad_base import AllocatorBadBase
from request_streams.base_request_stream_dist import BaseRequestStreamDist
from request_streams.ff_bad import FFBad
from request_streams.wf_good import WFGood
import random

class MixedFFBadWFGood(AllocatorBadBase):
    
    def _create_trajectory(self):
        self.ff_bad = FFBad(page_size=self.page_size)
        self.wf_good = WFGood(page_size=self.page_size)
        this_traj = []
        
        ff_bad_traj = self.ff_bad._create_trajectory()
        wf_good_traj = self.wf_good._create_trajectory()
        # #free all
        # this_traj += ff_bad_traj
        
        # for req in ff_bad_traj:
        #     if req[0] == 1:
        #         cnt += 1
        #     elif req[0] == "free":
        #         cnt -= 1
        # for i in range(cnt):
        #     this_traj.append(("free",0,0))
                
        # this_traj += wf_good_traj
        # return this_traj
        all_adversarial = [ff_bad_traj, wf_good_traj]

        this_traj = []
        order = [random.choice([0,1]), random.choice([0,1])]
        this_traj += all_adversarial[order[0]]
        
        cnt = 0
        #free everything
        for req in this_traj:
            if req[0] == 1:
                cnt += 1
            elif req[0] == "free":
                cnt -= 1
        for i in range(cnt):
            this_traj.append(("free",0,0))

        this_traj += all_adversarial[order[1]]

        #choose random amount between 1 and 5
        # num_rand_start = random.choice(range(1,6))
        # for i in range(num_rand_start):
        #     this_traj.append(super(AllocatorBadBase, self).get_next_req())
        #print(this_traj)

        return this_traj
    