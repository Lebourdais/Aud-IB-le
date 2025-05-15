import torch 
import torch.nn as nn

class SAE(nn.Module):
    """
    Sparse Autoencoder for dictionary learning in the latent space
    """
    def __init__(self, input_dim, sae_dim,sparsity=0.05, method='top-k'):
        super(SAE, self).__init__()
        self.input_dim = input_dim
        self.sae_dim = sae_dim
        self.sparsity = sparsity
        self.method = method

        # Encoder and decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, sae_dim, bias=True),
            nn.ReLU())
        self.decoder = nn.Linear(sae_dim, input_dim, bias=False)


    def forward(self, x):
        # Encoder
        z = self.encoder(x)

        # Apply sparsity constraint
        if self.method == 'top-k':
            k = int(self.sparsity * self.sae_dim) 
            _, indices = torch.topk(z, k, dim=-1) 
            mask = torch.zeros_like(z,dtype=z.dtype)
            mask.scatter_(2, indices, torch.ones_like(z, dtype=z.dtype))
            z = z * mask
        elif self.method == "archetypal":
            pass
        elif self.method == "jump_relu":
            pass

        # Decoder
        x_reconstructed = self.decoder(z)

        return x_reconstructed, z