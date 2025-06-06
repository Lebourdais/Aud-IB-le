import torch
import torch.nn as nn
import torch.nn.functional as F
from audIBle.nn.sae import SAE
from audIBle.nn.autoencoders import SpecAE
from audIBle.nn.Tcn import TCN
from audIBle.nn.utils import freeze_model

class SparseClassifier(nn.Module):

    def __init__(self,
                 n_classes: int,
                 autoencoder: nn.Module,
                 hidden_dim: int = 128,
                 sae_dim: int = 256,
                 sparsity: float = 0.05,
                 method: str = "top-k",
                 freeze_autoencoder: bool = False,
                 decode_sae_out: bool = False):
        super(SparseClassifier,self).__init__()

        self.freeze_autoencoder = freeze_autoencoder
        if freeze_autoencoder:
            autoencoder = freeze_model(autoencoder, eval_mode=True)

        self.audio_ae = autoencoder

        self.sae = SAE(input_dim=hidden_dim,
                       sae_dim=sae_dim,
                       sparsity=sparsity,
                       method=method)
        self.classif_head = nn.Linear(sae_dim,n_classes,bias=False)
        self.pool = TemporalAttentionPooling(hidden_dim=sae_dim)
        self.decode_sae_out = decode_sae_out

    def forward(self, wav):
        """
        Forward audio through the model. 
        Returns: 
            `y_hat`: classification logits
            `x_hat`: decoded audio
            `enc_audio_hat`: SAE output in the latent space
            `enc_audio`: SAE input (for MSE loss)
        """
        # audo-encode the audio
        # returns: reconstructed spectrogram, input spectrogram, latent rep, enhanced latent rep (unused for now)
        if self.freeze_autoencoder:
            with torch.no_grad():
                # spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
                spec, hidden = self.audio_ae.encode(wav)
                if not self.decode_sae_out:
                    spec_reconstruct = self.audio_ae.decode(hidden)
        else:
            # spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
            spec, hidden = self.audio_ae.encode(wav)
            if not self.decode_sae_out:
                spec_reconstruct = self.audio_ae.decode(hidden)

        hidden = hidden.permute(0,2,1)    
        # print(A.shape)
        # sparse autoencoder applied to the latent representation of the input
        b, h, t = hidden.shape
        hidden_reconstruct, sparse_latent = self.sae(hidden)

        if self.decode_sae_out:
            if self.freeze_autoencoder:
                with torch.no_grad():
                    spec_reconstruct = self.audio_ae.decode(hidden_reconstruct.permute(0,2,1))
            else:
                spec_reconstruct = self.audio_ae.decode(hidden_reconstruct.permute(0,2,1))

        # temporal attentive pooling on the SAE latent representation before classifying
        Z_pooled = self.pool(sparse_latent)

        # classify from the sparse latent representation
        y_hat = self.classif_head(Z_pooled)

        return y_hat, spec_reconstruct, spec, hidden, hidden_reconstruct
    
    def forward_return_all(self, wav):
        # if self.freeze_autoencoder:
        #     with torch.no_grad():
        #         spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
        # else:
        #     spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
        # hidden = hidden.permute(0,2,1)    
        # # print(A.shape)
        # # sparse autoencoder applied to the latent representation of the input
        # b, h, t = hidden.shape
        # hidden_reconstruct, sparse_latent = self.sae(hidden)

        # # temporal attentive pooling on the SAE latent representation before classifying
        # Z_pooled, att_weights = self.pool(sparse_latent, return_att_weights=True)

        # # classify from the sparse latent representation
        # y_hat = self.classif_head(Z_pooled)

        if self.freeze_autoencoder:
            with torch.no_grad():
                # spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
                spec, hidden = self.audio_ae.encode(wav)
                if not self.decode_sae_out:
                    spec_reconstruct = self.audio_ae.decode(hidden)
        else:
            # spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
            spec, hidden = self.audio_ae.encode(wav)
            if not self.decode_sae_out:
                spec_reconstruct = self.audio_ae.decode(hidden)

        hidden = hidden.permute(0,2,1)    
        # print(A.shape)
        # sparse autoencoder applied to the latent representation of the input
        b, h, t = hidden.shape
        hidden_reconstruct, sparse_latent = self.sae(hidden)

        if self.decode_sae_out:
            if self.freeze_autoencoder:
                with torch.no_grad():
                    spec_reconstruct = self.audio_ae.decode(hidden_reconstruct.permute(0,2,1))
            else:
                spec_reconstruct = self.audio_ae.decode(hidden_reconstruct.permute(0,2,1))

        # temporal attentive pooling on the SAE latent representation before classifying
        Z_pooled, att_weights = self.pool(sparse_latent, return_att_weights=True)

        # classify from the sparse latent representation
        y_hat = self.classif_head(Z_pooled)

        all_rep = {
            "spec": spec,
            "spec_hat": spec_reconstruct, 
            "ae_hidden": hidden,
            "ae_hidden_hat": hidden_reconstruct,
            "sparse_latent": sparse_latent,
            "sparse_latent_pooled": Z_pooled,
            "attention_weights": att_weights,
            "logits": y_hat
        }

        return all_rep

    def get_stft(self,wav):
        return self.audio_ae.get_stft(wav)
        

class SparseAEClassifier(nn.Module):

    def __init__(self,
                 n_classes: int,
                 autoencoder: nn.Module,
                 hidden_dim: int = 128,
                 freeze_autoencoder: bool = True,
                 use_seq_modeling: bool = False,
                 tcn_bn_dim: int = 128,
                 tcn_hid_dim: int = 64,
                 tcn_n_repeat: int = 1,
                 tcn_n_block: int = 3,
                 pool_type: str = "att",
                 **kwargs):
        """Classifier using a sparse audio autoencoder.
        This AE is trained with top-k over the latent space to reconstruct spectrograms.

        Args:
            n_classes (int): number of classes to predict
            autoencoder (nn.Module): sparse audio autoencoder 
            hidden_dim (int, optional): Dimension of the latent space of the AE. Defaults to 128.
            freeze_autoencoder (bool, optional): Keeps the autoencoder frozen if True. Defaults to False.
        """
        super(SparseAEClassifier,self).__init__()

        self.freeze_autoencoder = freeze_autoencoder
        if freeze_autoencoder:
            autoencoder = freeze_model(autoencoder, eval_mode=True)
        self.audio_ae = autoencoder
        self.use_seq_modeling = use_seq_modeling
        if use_seq_modeling:
            self.seq_model = TCN(in_chan=hidden_dim, 
                                 n_src=1, 
                                 out_chan=n_classes, 
                                 n_blocks=tcn_n_block, 
                                 n_repeats=tcn_n_repeat,
                                 hid_chan=tcn_hid_dim, 
                                 bn_chan=tcn_bn_dim)
            self.classif_head = nn.Linear(n_classes,n_classes,bias=False)

        if pool_type == "att":
            self.pool = TemporalAttentionPooling(hidden_dim=hidden_dim)
            self.classif_head = nn.Linear(hidden_dim,n_classes,bias=False)
        elif pool_type == "avg":
            self.classif_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # Temporal pooling
                nn.Flatten(),
                nn.Linear(hidden_dim, hidden_dim//4),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim//4, n_classes)
        )
        self.pool_type = pool_type

    def forward(self, wav):
        """
        Forward audio through the model. 
        Returns: 
            `y_hat`: classification logits
            `x_hat`: decoded audio
            `enc_audio_hat`: SAE output in the latent space
            `enc_audio`: SAE input (for MSE loss)
        """
        # audo-encode the audio
        # returns: reconstructed spectrogram, input spectrogram, latent rep, enhanced latent rep (unused for now)
        if self.freeze_autoencoder:
            with torch.no_grad():
                # spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
                spec, hidden = self.audio_ae.encode(wav)
                spec_reconstruct = self.audio_ae.decode(hidden)
        else:
            # spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
            spec, hidden = self.audio_ae.encode(wav)
            spec_reconstruct = self.audio_ae.decode(hidden)

        # sparse autoencoder applied to the latent representation of the input

        if self.use_seq_modeling:
            hidden_seq = self.seq_model(hidden)
            #hidden_seq = hidden_seq.transpose(1,2)
            y_hat = self.classif_head(hidden_seq.mean(dim=-1))
        
        
        else:
            # temporal attentive pooling on the SAE latent representation before classifying    
            if self.pool_type == "avg":
                y_hat = self.classif_head(hidden)
            else:
                hidden = hidden.permute(0,2,1) #(batch,frame,hidden)
                Z_pooled = self.pool(hidden)
                # classify from the sparse latent representation
                y_hat = self.classif_head(Z_pooled)

        return y_hat, spec_reconstruct, spec, hidden
    
    def forward_return_all(self, wav):
        # if self.freeze_autoencoder:
        #     with torch.no_grad():
        #         spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
        # else:
        #     spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
        # hidden = hidden.permute(0,2,1)    
        # # print(A.shape)
        # # sparse autoencoder applied to the latent representation of the input
        # b, h, t = hidden.shape
        # hidden_reconstruct, sparse_latent = self.sae(hidden)

        # # temporal attentive pooling on the SAE latent representation before classifying
        # Z_pooled, att_weights = self.pool(sparse_latent, return_att_weights=True)

        # # classify from the sparse latent representation
        # y_hat = self.classif_head(Z_pooled)

        if self.freeze_autoencoder:
            with torch.no_grad():
                # spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
                spec, hidden = self.audio_ae.encode(wav)
                spec_reconstruct = self.audio_ae.decode(hidden)
        else:
            # spec_reconstruct, spec, hidden, _ = self.audio_ae(wav) 
            spec, hidden = self.audio_ae.encode(wav)
            spec_reconstruct = self.audio_ae.decode(hidden)

        hidden = hidden.permute(0,2,1)    
        # print(A.shape)
        # sparse autoencoder applied to the latent representation of the input

        # temporal attentive pooling on the SAE latent representation before classifying
        Z_pooled, attention_weights = self.pool(hidden, return_att_weights=True)

        # classify from the sparse latent representation
        y_hat = self.classif_head(Z_pooled)

        all_rep = {
            "spec": spec,
            "spec_hat": spec_reconstruct, 
            "ae_hidden": hidden,
            "sparse_latent_pooled": Z_pooled,
            "attention_weights": attention_weights,
            "logits": y_hat
        }

        return all_rep

    def get_stft(self,wav):
        return self.audio_ae.get_stft(wav)


class TemporalAttentionPooling(nn.Module):
    """
    Module d'attention pour agréger la dimension temporelle de manière
    adaptative en donnant plus de poids aux trames temporelles importantes.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttentionPooling, self).__init__()
        
        # Mécanisme d'attention temporelle
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, return_att_weights=False):
        # x: (batch, time, hidden_dim)
        
        # Calculer les scores d'attention
        attn_scores = self.attention(x)  # (batch, time, 1)
        
        # Normaliser avec softmax
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, time, 1)
        
        # Somme pondérée
        context = torch.bmm(x.transpose(1, 2), attn_weights)  # (batch, hidden_dim, 1)
        context = context.squeeze(2)  # (batch, hidden_dim)
        if return_att_weights:
            return context, attn_weights
        return context

class AveragePooling(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
    def forward(self,x):
        return x.mean(dim=self.dim)

def test():

    print("Test audio classifier ...................")

    n_batch, n_chan, n_sample = (64, 1, 32000)

    x = torch.randn((n_batch, n_chan, n_sample))
    ae = SpecAE(hid_dims=[16,32,64,128], latent_dim=256)
    sparse_classif = SparseClassifier(n_classes=10,
                                      autoencoder=ae,
                                      hidden_dim=256,
                                      sae_dim=1024,)

    y_hat, x_hat, A_hat, A, Z = sparse_classif(x)

    print(f"{y_hat.shape=}")
    print(f"{x_hat.shape=}")
    print(f"{A_hat.shape=}")
    print(f"{A.shape=}")
    print(f"{Z.shape=}")

    print("Test done!")

if __name__ == "__main__":
    test()
