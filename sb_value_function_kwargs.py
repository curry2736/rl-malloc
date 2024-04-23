import gym
import torch

fc_policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[32, 32])