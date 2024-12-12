import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        # We need to know the size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "Grid":
                extractors[key] = nn.Sequential(
                    nn.Conv2d(subspace.shape[0], 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Flatten()
                )
                conv_output_size = subspace.shape[1] * subspace.shape[2]
                total_concat_size += conv_output_size
            elif key == "Stresses":
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )
                total_concat_size += 32

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # converted_obs = {
        #     key: th.tensor(value).unsqueeze(0) for key, value in observations.items()
        # }
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            input_tensor = observations[key].to(next(extractor.parameters()).device)
            if key == "Grid":
                # print(input_tensor.shape)
                # print(input_tensor)
                # Normalize the third and fourth channels of the grid
                input_tensor[:, 2, :, :] = (input_tensor[:, 2, :, :] + 80) / 160
                input_tensor[:, 3, :, :] = (input_tensor[:, 3, :, :] + 80) / 160
                # Multiply the first and second channels by 255
            encoded_tensor_list.append(extractor(input_tensor))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)