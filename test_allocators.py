from allocators.best_fit_allocator import BestFitAllocator
from allocators.worst_fit_allocator import WorstFitAllocator
from allocators.first_fit_allocator import FirstFitAllocator
from environments.single_state_env import SingleStateEnv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

page_size = 100


env = SingleStateEnv(page_size=page_size, allocator="wf_good")

num_episodes = 100
all_returns = {}
allocators = [BestFitAllocator(), WorstFitAllocator(), FirstFitAllocator()]
for allocator in allocators:
    curr_allocator_return = []
    for ep in tqdm(range(num_episodes)):
        s, done = env.reset()
        r = -1

        count = 0
        rewards = []
        while not done:
            #ac is mem_addr_or_amt
            curr_page = s["pages"][0]
            if s["rq"][0] == 0:
                ac = s["rq"][1]
            else:
                ac = allocator.handle_alloc_req(s["pages"], s["rq"][1])[1]

            s,r,done = env.step([s["rq"][0], ac])
            rewards.append(r)
            # print("Current page's resulting free list: ", curr_page.free_list, "Current page's resulting allocated list: ", curr_page.allocated_list)
            # print("----------------------------------")
            count += 1
        curr_allocator_return.append(sum(rewards))
    all_returns[type(allocator).__name__] = curr_allocator_return

for name, returns in all_returns.items():
    avg = np.average(returns[-100:])
    std = np.std(returns[-100:])
    print(f"{name} 95% conf int: {avg} +- {std}")
    plt.figure()
    plt.plot(returns, label=name)
    plt.hlines(avg, 0, num_episodes, colors='r', linestyles='dashed')
    plt.title(f"{name} returns")
    plt.xlabel("episode")
    plt.ylabel("return")
