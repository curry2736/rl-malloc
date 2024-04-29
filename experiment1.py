from sb_value_function_kwargs import fc_policy_kwargs, cnn_policy_kwargs
from environments.stable_baselines_env import StableBaselineEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats



def mean_confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    mean, std_err = np.mean(data), np.std(data) / np.sqrt(n)
    interval = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - interval, mean + interval, mean


allocator_name = "dist"
page_size=10
env = StableBaselineEnv(allocator=allocator_name, page_size=page_size, high_level_actions=False)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
model_name = "PPO_MLP_1000000"
model = PPO.load(f"models/{model_name}", env=env, device="cuda")



#model attributes
print(model.__dict__)

# timesteps=1000000
# model.learn(total_timesteps=timesteps, log_interval=4, progress_bar=True)
#model.save(f"PPO_MLP_{timesteps}")


num_actions = page_size

obs, info = env.reset()
i = 0
rewards = []
curr_reward = 0
action_counts = [0] * num_actions
pbar = tqdm(total = 11)
print("evaluation!")
while i < 100:
    action, _states = model.predict(obs, deterministic=True)
    action_counts[action] += 1
    obs, reward, terminated, truncated, info = env.step(action)
    curr_reward += reward
    if terminated or truncated or curr_reward < -200:
        if curr_reward < -200:
            print("failed run iteration ", i)
        pbar.update(1)
        obs, info = env.reset()
        i += 1
        rewards.append(curr_reward)
        curr_reward = 0


model_conf_int = mean_confidence_interval(np.array(rewards))
model_conf_int = np.array(model_conf_int)
print(model_conf_int)

#switch to high level actions
env = StableBaselineEnv(allocator=allocator_name, page_size=page_size, high_level_actions=True)

conf_ints_baselines = []
for allocator in tqdm(range(3)):
    obs, info = env.reset()
    i = 0
    rewards = []
    curr_reward = 0
    while i < 100:
        
        obs, reward, terminated, truncated, info = env.step(allocator)
        if reward != .1 and reward != 0:
            print(reward)
        curr_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
            i += 1
            rewards.append(curr_reward)
            curr_reward = 0


    conf_ints_baselines.append(mean_confidence_interval(np.array(rewards)))

print(conf_ints_baselines)
conf_ints_baselines = np.array(conf_ints_baselines)

# #create a bar graph with the confidence intervals
all_conf_ints = np.vstack((model_conf_int, conf_ints_baselines))

plt.bar(x=np.arange(4), height=all_conf_ints[:,2], yerr=all_conf_ints[:, 2] - all_conf_ints[:, 0])
plt.title(f"Average reward over 100 episodes for {model_name} on \"{allocator_name}\" \n alloc sequences \n with 95% confidence intervals")
plt.xticks(np.arange(4), ["our policy", "best fit", "worst fit", "first fit"])
plt.savefig(f"results/{allocator_name}_{model_name}_100_episodes.png",  bbox_inches = "tight")
plt.show()
