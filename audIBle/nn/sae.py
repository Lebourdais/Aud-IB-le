import torch 
import torch.nn as nn
from audIBle.nn.pretrained_models import WavLMEncoder, HuBERTEncoder, BEATsEncoder, ASTEncoder, MERTEncoder
from typing import Dict, List, Tuple, Optional, Union

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
        if self.sparsity > 0:
            if self.method == 'top-k':
                k = int((1-self.sparsity) * self.sae_dim) 
                _, indices = torch.topk(z, k, dim=-1) 
                mask = torch.zeros_like(z,dtype=z.dtype)
                mask.scatter_(2, indices, torch.ones_like(z, dtype=z.dtype))
                z = z * mask
            elif self.method == "jump_relu":
                pass

        # Decoder
        x_reconstructed = self.decoder(z)

        return x_reconstructed, z

class MeanPooling(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim=dim

    def forward(self, x):
        return x.mean(dim=self.dim,keepdim=True)

class SaeSslWrapper(nn.Module):

    def __init__(self,
                 encoder_type: str,
                 sae_method: str,
                 sae_dim: int,
                 sparsity: float,
                 freeze: bool = True,
                 layer_indices: list[int] = [-1],
                 pooling_method: str = None,
                 ):
        super().__init__()

        if encoder_type.upper() == "WAVLM":
            model_name='microsoft/wavlm-base-plus'
            self.encoder = WavLMEncoder(model_name=model_name, freeze_encoder=freeze)        
        elif encoder_type.upper() == "HUBERT":
            model_name = "facebook/hubert-base-ls960"
            self.encoder = HuBERTEncoder(model_name=model_name, freeze_encoder=freeze)
        elif encoder_type.upper() == "BEATS":
            model_name = 'microsoft/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2'
            self.encoder = BEATsEncoder(model_name=model_name, freeze_encoder=freeze)
        elif encoder_type.upper() == "AST":
            model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
            self.encoder = ASTEncoder(model_name=model_name, freeze_encoder=freeze)      
        elif encoder_type.upper() == "MERT":
            model_name = "m-a-p/MERT-v1-95M"
            self.encoder = MERTEncoder(model_name=model_name, freeze_encoder=freeze)    
        else:
            raise Exception(f"!!! No audio encoder can be found with {encoder_type=} !!!")


        # check sparsity parameter
        print(f"!!! Sparsity = {sparsity}")
        # One SAE for each selected hidden state
        n_sae = len(layer_indices) + 1 # include the last hidden state
        sae_set = []
        hid_dim = self.encoder.hidden_size
        for ii in range(n_sae):
            sae_set.append(SAE(input_dim=hid_dim, sae_dim=sae_dim, sparsity=sparsity, method=sae_method))

        self.saes = nn.ModuleList(sae_set)
        
        self.encoder_type = encoder_type
        self.layer_indices = layer_indices
        
        if pooling_method is not None:
            if pooling_method == "mean":
                self.pool = MeanPooling(dim=1)
            else:
                self.pool = None

    def forward(self, x):
        outputs = self.encoder(audio=x, output_hidden_states=True, layer_indices=self.layer_indices)
        out_cat = outputs["hidden_states"]
        out_cat.append(outputs["last_hidden_state"])

        sae_latent = []
        hidden_reconstruct = []
        hidden = []
        for ii, hid_rep in enumerate(out_cat):
            if self.pool is not None:
                hid_rep = self.pool(hid_rep)
            hid_hat, z_sparse = self.saes[ii](hid_rep)
            sae_latent.append(z_sparse)
            hidden_reconstruct.append(hid_hat)
            hidden.append(hid_rep)

        return hidden, sae_latent, hidden_reconstruct
