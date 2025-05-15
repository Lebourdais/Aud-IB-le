import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeLayer(nn.Module):
    def __init__(self, n_prototypes, n_chan_latent, n_freq_latent, n_frames_latent, distance='euclidean', use_weighted_sum=True):
        super(PrototypeLayer, self).__init__()
        self.n_prototypes = n_prototypes
        self.distance = distance
        self.use_weighted_sum = use_weighted_sum
        self.kernel = nn.Parameter(torch.rand(self.n_prototypes, n_chan_latent, n_freq_latent, n_frames_latent), requires_grad=True)
        # print(f"PrototypeLayer kernel shape: {self.kernel.shape}")

    def forward(self, x):
        # x: (batch_size, T, H, W)
        # if self.kernel is None:
        #     self.kernel = nn.Parameter(torch.randn(
        #         self.n_prototypes, *x.shape[1:]), requires_grad=True).to(x.device)  # shape: (P, T, H, W)
        x_reshape = x.unsqueeze(1)
        kernel_reshape = self.kernel.unsqueeze(0)
        kernel_reshape_normalized = F.normalize(kernel_reshape, p=2)
        full_dist = (x_reshape - kernel_reshape_normalized).pow(2)
        dist = full_dist.sum(dim=-1).mean(dim=2)
        return dist

class WeightedSum(nn.Module):
    def __init__(self, D3=False, n_prototypes=10, n_freq_latent=64):
        super(WeightedSum, self).__init__()
        self.D3 = D3
        shape = (n_prototypes, n_freq_latent)  # (P, T)
        init_val = 1.0 / n_freq_latent
        self.kernel = nn.Parameter(torch.full(shape, init_val), requires_grad=True)

    def forward(self, x):
        # x: (B, P, T) or (B, P, H, W)
        # print(f"weighting sum layer input shape: {x.shape}")
        # if self.kernel is None:
        #     if self.D3:
        #         shape = (x.size(1), x.size(2), x.size(3))  # (P, H, W)
        #     else:
        #         shape = (x.size(1), x.size(2))  # (P, T)
        #     init_val = 1.0 / shape[-1]
        #     self.kernel = nn.Parameter(torch.full(shape, init_val)).to(x.device)  # shape: (P, T) or (P, H, W)
        # print(f"{self.kernel.shape=}")

        # Normalize the kernel along the last axis
        norm_kernel = F.normalize(self.kernel, p=2)
        if self.D3:
            return torch.sum(x * self.kernel.unsqueeze(0), dim=-1)  # (B, P, H)
        else:
            # print(f"output shape WS: {torch.sum(x * norm_kernel.unsqueeze(0), dim=-1).shape}")
            return torch.sum(x * norm_kernel.unsqueeze(0), dim=-1)  # (B, P)
