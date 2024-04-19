from policies.state_value_nn import NNValueFn
from policies.nn_fit import NNFitPolicy
from environments.single_state_env import SingleStateEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

#Based on sutton and barto pseudo code
def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:NNFitPolicy,
    n:int,
    alpha:float,
    V:NNValueFn,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """

    all_returns = []
    action_count_per_ep = []
    for ep in range(num_episode):
        cap_T = float('inf')
        tau = -1
        t = 0

        rewards = []
        states = []
        state,  done = env.reset()
        acc_r = 0.
        states.append(state['pages'][0])
        rewards.append(acc_r)
        action_count = [0] * 3
        while tau < cap_T:
            if t < cap_T:
                rq = state["rq"]
                if rq[0] == 0:
                    state, r, done, = env.step([rq[0], rq[1]])
                else:
                    a = pi.action(state)
                    action_idx = a[2]
                    action_count[action_idx] += 1
                    a = (a[0], a[1])
                    state, r, done,  = env.step(a)

                states.append(state['pages'][0])
                rewards.append(r)
                if done:
                    cap_T = t + 1
                
            tau = t - n + 1
            if tau >= 0:    
                sum_end = min(tau + n, cap_T)
                i = tau + 1
                G = 0
                while i <= sum_end:
                    G += (gamma ** (i - tau - 1) * rewards[i])
                    i += 1
                if tau + n < cap_T:
                    G = G + (gamma ** (n)) * V(states[tau + n])
                #print(f"episode {ep}: G = {G}")
                V.update(alpha, G, states[tau])
                #break
            t += 1
        print(f"episode {ep}: {sum(rewards)}")
        all_returns.append(sum(rewards))
        action_count_per_ep.append(action_count)

    action_count_per_ep = np.array(action_count_per_ep)
    #plot returns for each episode
    return all_returns, action_count_per_ep


gamma = 1.
page_size = 1000
num_episodes = 1000
env = SingleStateEnv(page_size=page_size)
V = NNValueFn(page_size)
policy = NNFitPolicy(V)

all_returns, action_count_per_ep = semi_gradient_n_step_td(env,1.,policy,10,0.001,V,num_episodes)