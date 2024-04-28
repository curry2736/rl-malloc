import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 3),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    


fc_policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[32, 32])

cnn_policy_kwargs = dict(features_extractor_class=CustomCNN, net_arch=[])
deep_cnn_policy_kwargs = dict(features_extractor_class=CustomCNN, net_arch=[128,64])

linear_policy_kwargs = dict(net_arch=[])



# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )

# model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
# model.learn(1000)