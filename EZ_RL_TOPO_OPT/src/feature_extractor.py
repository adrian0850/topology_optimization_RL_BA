import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that combines convolutional and linear layers to 
    process different parts of the observation space.
    Args:
        observation_space (gym.spaces.Dict): The observation space containing 
        different subspaces.
    Attributes:
        feature_maps (dict): Dictionary to store feature maps for visualization.
        extractors (nn.ModuleDict): Dictionary of feature extractors for each subspace.
        _features_dim (int): The dimension of the concatenated features.
    Methods:
        get_activation(name):
            Registers a forward hook to capture the output of a layer.
        forward(observations) -> th.Tensor:
            Forward pass to extract and concatenate features from the observation space.
        visualize_feature_maps():
            Visualizes the feature maps stored in the feature_maps attribute.
    """
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        self.feature_maps = {}

        extractors = {}
        total_concat_size = 0

        # We need to know the size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "Grid":
                layers = [
                    nn.Conv2d(subspace.shape[0], 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Flatten()
                ]
                extractors[key] = nn.Sequential(*layers)
                conv_output_size = subspace.shape[1] * subspace.shape[2]
                total_concat_size += conv_output_size

                for i, layer in enumerate(layers):
                    if isinstance(layer, nn.Conv2d):
                        layer.register_forward_hook(self.get_activation(f"{key}_conv{i}"))

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

    def get_activation(self, name):
        def hook(model, input, output):
            self.feature_maps[name] = output.detach()
        return hook

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

    def visualize_feature_maps(self):
        for name, feature_map in self.feature_maps.items():
            num_channels = feature_map.shape[1]
            size = feature_map.shape[2]
            _, axes = plt.subplots(num_channels, 1, figsize=(2, num_channels * 2))
            if num_channels == 1:
                axes = [axes]  # Ensure axes is always a list
            for i in range(num_channels):
                ax = axes[i]
                ax.imshow(feature_map[0, i].cpu().numpy(), cmap='viridis')
                ax.axis('off')
            plt.suptitle(f"Feature maps from {name}")
            plt.show()
