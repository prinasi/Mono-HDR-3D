import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class LightKnowledgeInspiredLdr2HdrConverter(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super(LightKnowledgeInspiredLdr2HdrConverter, self).__init__()
        
        self.comm = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.x_net = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
            # nn.LeakyReLU(),
            nn.ReLU(),
        )
        self.r_net = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

        self.y_net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        
        self.post_process = nn.Sequential(
            # nn.Linear(1, 1),
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


class LightKnowledgeInspiredHdr2LdrConverter(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super(LightKnowledgeInspiredHdr2LdrConverter, self).__init__()
        warnings.warn("Using the light KIIC now...")

        self.comm = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.x_net = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.ReLU(),
        )

        self.r_net = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
        self.post_process = nn.Sequential(
            nn.Sigmoid()
        )

    def _forward(self, hdr: torch.Tensor) -> torch.Tensor:
        unsqueeze = False
        if hdr.ndim == 4:
            hdr = hdr[0]
            unsqueeze = True

        assert hdr.ndim == 3, f"hdr should be a 3-d tensor, but got {hdr.shape}"
        hdr = hdr.permute(1, 2, 0)  # [800, 800, 3]

        comm = self.comm(hdr)       # [800, 800, 128]
        x_out = self.x_net(comm)    # [800, 800, 3]
        r_out = self.r_net(comm)    # [800, 800, 3]
        ldr = x_out + r_out
        ldr = self.post_process(ldr)
        ldr = ldr.permute(2, 0, 1) 
        
        if unsqueeze:
            ldr = ldr[None]
        return ldr
    
    def forward(self, hdr: torch.Tensor) -> torch.Tensor:
        return cp.checkpoint(self._forward, hdr)


class SimpleHdr2LdrConverter(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super(SimpleHdr2LdrConverter, self).__init__()
        warnings.warn("Using the simple Hdr2Ldr converter now...")

        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def _forward(self, hdr: torch.Tensor) -> torch.Tensor:
        unsqueeze = False
        if hdr.ndim == 4:
            hdr = hdr[0]
            unsqueeze = True

        assert hdr.ndim == 3, f"hdr should be a 3-d tensor, but got {hdr.shape}"
        hdr = hdr.permute(1, 2, 0)  # [800, 800, 3]
        ldr = self.net(hdr)         # [800, 800, 3]
        ldr = ldr.permute(2, 0, 1) 
        
        if unsqueeze:
            ldr = ldr[None]
        return ldr
    
    def forward(self, hdr: torch.Tensor) -> torch.Tensor:
        return cp.checkpoint(self._forward, hdr)


class LightKnowledgeInspiredLdr2HdrConverterForAblation(nn.Module):
    def __init__(self, hidden_dim: int = 128, layers: int = 0, with_no_hidden: bool = False):
        super(LightKnowledgeInspiredLdr2HdrConverterForAblation, self).__init__()
        
        if with_no_hidden:
            warnings.warn("A single 3x3 linear layer is used now...")
            assert layers == 0, "The number of layers should be 0 when with_no_hidden is True"
            self.net = nn.Linear(1, 1)
        else:
            net = [
                # nn.Linear(1, hidden_dim),
                # nn.ReLU(inplace=True)
            ]

            for i in range(layers):
                net.extend([nn.Linear(1, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1), nn.ReLU(inplace=True)])
                if i == layers - 1:
                    net.pop()

            # net.append(nn.Linear(hidden_dim, 1))
            self.net = nn.Sequential(*net)
        self.post_process = nn.Softplus()

    def _forward(self, ldr: torch.Tensor) -> torch.Tensor:
        hdr = self.net(ldr)
        return self.post_process(hdr)
    
    def forward(self, ldr: torch.Tensor) -> torch.Tensor:
        return cp.checkpoint(self._forward, ldr)
