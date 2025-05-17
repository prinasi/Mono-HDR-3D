import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import warnings


class LightKnowledgeInspiredLdr2HdrConverter(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super(LightKnowledgeInspiredLdr2HdrConverter, self).__init__()
        warnings.warn("Using the light KICC now...")

        self.comm = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.x_net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )
        self.r_net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

        self.y_net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.post_process = nn.Sequential(
            nn.Softplus()
        )

    def _forward(self, ldr: torch.Tensor) -> torch.Tensor:
        comm = self.comm(ldr)
        x_out = self.x_net(comm)
        r_out = self.r_net(comm)
        y_out = self.y_net(comm)
        hdr = ldr + x_out * r_out + y_out
        return self.post_process(hdr)

    def forward(self, ldr: torch.Tensor) -> torch.Tensor:
        return cp.checkpoint(self._forward, ldr)
    

class LightKnowledgeInspiredNetwork(nn.ModuleList):
    def __init__(self, hidden_channels: int = 128, num_dc_layers: int = 3, num_res_layers: int = 3):
        super(LightKnowledgeInspiredNetwork, self).__init__()

        warnings.warn("Using the light KIN now...")

        self.dense_connect_block = [
            nn.Linear(3, 3),
            nn.ReLU(inplace=True)
        ]
        self.residual_block = [
            nn.Linear(3, 3),
            nn.Tanh()
        ]
        self.dense_connect_layers = nn.Sequential(*(self.dense_connect_block * num_dc_layers))
        self.residual_layers = nn.Sequential(*(self.residual_block * num_res_layers))

        self.pre_process = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(hidden_channels, 3),
            nn.ReLU(inplace=True),
            # nn.Tanh()
        )

        self.post_process = nn.Sequential(
            # nn.Conv2d(3, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def _forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [1024, 3]
        features = self.pre_process(image)
        y1 = self.dense_connect_layers(features)
        y2 = self.residual_layers(features)
        img = self.post_process(y1 + y2)
        return img
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return cp.checkpoint(self._forward, image)
