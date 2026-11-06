# --------------------------------------------------------
# This code is borrowed from https://github.com/facebookresearch/deep_bisim4control/blob/main/transition_model.py
# --------------------------------------------------------

import torch
import torch.nn as nn


class DeterministicTransitionModel(nn.Module):
    def __init__(self, encoder_feature_dim, action_shape, layer_width):
        super().__init__()
        self.fc = nn.Linear(encoder_feature_dim + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        return self.fc_mu(x)
