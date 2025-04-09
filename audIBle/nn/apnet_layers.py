import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeLayer(nn.Module):
    def __init__(self, n_prototypes, distance='euclidean', use_weighted_sum=True):
        super(PrototypeLayer, self).__init__()
        self.n_prototypes = n_prototypes
        self.distance = distance
        self.use_weighted_sum = use_weighted_sum
        self.kernel = None  # will be initialized in `forward` based on input

    def forward(self, x):
        # x: (batch_size, T, H, W)
        if self.kernel is None:
            self.kernel = nn.Parameter(torch.randn(
                self.n_prototypes, *x.shape[1:]), requires_grad=True).to(x.device)  # shape: (P, T, H, W)

        x_expanded = x.unsqueeze(1)  # shape: (B, 1, T, H, W)
        kernel_expanded = self.kernel.unsqueeze(0)  # shape: (1, P, T, H, W)

        if self.distance == 'euclidean':
            if self.use_weighted_sum:
                axis = (2, 4)
            else:
                axis = (2, 3, 4)
            dist = torch.sum((x_expanded - kernel_expanded) ** 2, dim=axis)
        
        elif self.distance == 'euclidean_patches':
            dist = torch.sum((x_expanded - kernel_expanded) ** 2, dim=-1)

        elif self.distance == 'cosine':
            if self.use_weighted_sum:
                axis = (2, 4)
            else:
                axis = (2, 3, 4)

            x_norm = F.normalize(x_expanded, p=2, dim=axis, eps=1e-12)
            kernel_norm = F.normalize(kernel_expanded, p=2, dim=axis, eps=1e-12)
            dist = -torch.sum(x_norm * kernel_norm, dim=axis)
        else:
            raise ValueError(f"Unknown distance type: {self.distance}")

        return dist

class WeightedSum(nn.Module):
    def __init__(self, D3=False):
        super(WeightedSum, self).__init__()
        self.D3 = D3
        self.kernel = None  # Initialized in `forward`

    def forward(self, x):
        # x: (B, P, T) or (B, P, H, W)
        if self.kernel is None:
            if self.D3:
                shape = (x.size(1), x.size(2), x.size(3))  # (P, H, W)
            else:
                shape = (x.size(1), x.size(2))  # (P, T)
            init_val = 1.0 / shape[-1]
            self.kernel = nn.Parameter(torch.full(shape, init_val)).to(x.device)  # shape: (P, T) or (P, H, W)

        # Normalize the kernel along the last axis
        norm_kernel = F.normalize(self.kernel, p=2, dim=-1)
        if self.D3:
            return torch.sum(x * norm_kernel.unsqueeze(0), dim=-1)  # (B, P, H)
        else:
            return torch.sum(x * norm_kernel.unsqueeze(0), dim=-1)  # (B, P)
