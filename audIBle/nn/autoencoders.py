import torch
import torch.nn as nn
import numpy as np
import torchaudio
from collections import OrderedDict
from typing import Union

class SpecAE(nn.Module):

    def __init__(self,
                 normalize_audio = True,
                 scale="lin",
                 n_fft=1024,
                 win_length=1024,
                 hop_length=256,
                 center=True, 
                 pad=0,
                 input_channels=1,
                 latent_dim=256,
                 hid_dims=[16, 32, 64, 128],
                 use_attention=False,
                 activation_slope=0.2,
                 attention_dim=64):
        """Spectrogram auto-encoder

        This module auto-encodes a spectrogram. It can later be used for a downstream task such as sound event classification, detection...

        Args:
            spec_kw (dict, optional): parameters of the spectrogram. Defaults to {"win_length": 1024, "hop_length": 256, "n_fft": 1024}.
            ae_kw (dict, optional): parameters of the encoder (see `AttentiveAE`). Defaults to {"input_channels":1, "hid_dim": [16, 32, 64, 128],"use_attention":False, "activation_slope": 0.2, "attention_dim": 64}.
        """
        super(SpecAE,self).__init__()
        self.enc_dec = AttentiveAE(input_channels=input_channels,
                                   latent_dim=latent_dim,
                                   hid_dims=hid_dims,
                                   use_attention=use_attention,
                                   activation_slope=activation_slope,
                                   attention_dim=attention_dim)
        
        self.spec = Spectrogram(scale=scale,
                 n_fft=n_fft,
                 win_length=win_length,
                 hop_length=hop_length,
                 center=center, 
                 pad=pad,)
        self.normalize_audio = normalize_audio

    def forward(self,x):
        """_summary_

        Args:
            x (torch.Tensor): raw audio waveform

        Returns:
            x_hat: reconstructed spectrogram
            x: original spectrogram
            z: latent representation of the spectrogram
            z_enh: enhanced latent space if attention is used in the AE (`None` otherwise)
        """
        if self.normalize_audio:
            x = self.normalize_wav(x)
        x = self.spec(x)
        x_hat, z, z_att, feat = self.enc_dec(x)
        return x_hat, x, z, z_att
    
    def decode(self,z):
        return self.enc_dec.decode(z)
    
    def encode(self,x):
        if self.normalize_audio:
            x = self.normalize_wav(x)
        x = self.spec(x)
        z, feat = self.enc_dec.encode(x)
        return x, z

    @staticmethod
    def normalize_wav(wav):
        energy = torch.sum(wav ** 2, dim=-1, keepdim=True)
        energy = torch.clamp(energy, min=1e-6)  # Avoid division by zero
        wav = wav / torch.sqrt(energy)
        return wav


class Spectrogram(nn.Module):
    def __init__(self, scale, n_fft, hop_length, win_length, center=False, pad=0):
        super(Spectrogram, self).__init__()
        self.scale=scale
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                      hop_length=hop_length,
                                                      win_length=win_length,
                                                      center=center,
                                                      pad=pad)
        self.eps = 1e-6

    def forward(self,x):
        x = self.spec(x)
        x = x.abs()
        if self.scale == "log":
            return torch.log(x+self.eps)
        elif self.scale == "log_biais":
            return torch.log(x+1)
        elif self.scale == "db":
            return 20 * torch.log10(x+self.eps)
        else:
            return x

class AttentiveAE(nn.Module):
    """
    Auto-encodeur qui compresse uniquement la dimension fréquentielle tout en
    préservant intégralement la dimension temporelle.
    
    Structure du spectrogramme: (batch, channels, fréquence, temps)
    """
    def __init__(self, 
                 input_channels=1, 
                 input_freq_dim=513,  # Dimension fréquentielle d'entrée explicite
                 hid_dims=[16, 32, 64, 128],
                 latent_dim=256,
                 use_attention=False,
                 activation_slope=0.2,
                 attention_dim=64):
        super(AttentiveAE, self).__init__()
        
        self.input_freq_dim = input_freq_dim
        # print(f"Input frequency dimension: {input_freq_dim}")
        # print(f"Hidden dimensions: {hid_dims}")
        self.latent_dim = hid_dims[-1]
        kernel_size = (5,3)
        padding = (2,1)
        stride = (2,1)
        
        # Stocker ces valeurs pour le décodeur
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # Calculer les dimensions des features après chaque couche d'encodage
        self.encoder_dims = [input_freq_dim]
        curr_dim = input_freq_dim
        for _ in range(len(hid_dims)):
            curr_dim = (curr_dim + 2*padding[0] - kernel_size[0]) // stride[0] + 1
            self.encoder_dims.append(curr_dim)
        
        self.final_freq_dim = self.encoder_dims[-1]
        # print(f"Encoder dimensions: {self.encoder_dims}")
        
        n_block = len(hid_dims)
        enc_layers = OrderedDict()
        for ii in range(n_block):
            enc_layers[f"block_{ii}"] = ConvBlock(input_channels=input_channels if ii == 0 else hid_dims[ii-1], 
                                                 output_channels=hid_dims[ii], 
                                                 kernel_size=kernel_size, 
                                                 stride=stride, 
                                                 padding=padding,
                                                 act_slope=activation_slope)
            
        self.encoder = nn.Sequential(enc_layers)
        
        # Projection linéaire pour transformer de (channels, freq_dim) à (latent_dim)
        self.freq_projection = nn.Sequential(
            nn.Conv2d(hid_dims[-1], latent_dim, kernel_size=(self.final_freq_dim, 1), padding=0),
            nn.ReLU()
        )
        
        # Projection inverse pour le décodeur
        self.freq_expansion = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hid_dims[-1], kernel_size=(self.final_freq_dim, 1), padding=0),
            nn.ReLU()
        )
        

        if use_attention:
            self.temporal_attention = TemporalAttention(self.latent_dim, attention_dim)
        
        dec_layers = OrderedDict()
        rev_hid_dim = list(reversed(hid_dims))
        
        # Le décodeur a besoin de connaître les dimensions de sortie exactes
        # à chaque étape pour reconstruire correctement
        for jj in range(n_block):
            if jj == n_block-1:
                out_channels = input_channels
                target_freq_dim = input_freq_dim  # Dimension fréquentielle d'origine
            else:
                out_channels = rev_hid_dim[jj+1]
                target_freq_dim = self.encoder_dims[n_block - jj - 1]
                
            dec_layers[f"block_{jj}"] = DeconvBlock(
                in_channels=rev_hid_dim[jj],
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                target_freq_dim=target_freq_dim,
                use_activation=jj != (n_block-1),
                act_slope=activation_slope
            )

        self.decoder = nn.Sequential(dec_layers)
        self.use_attention = use_attention
        
        # Vérifier que la dernière étape du décodeur produit la bonne dimension fréquentielle
        # print(f"Target output frequency dimension: {input_freq_dim}")
    

    def encode(self, x):
        # Encodage - compression en fréquence
        features = self.encoder(x)
        # print(f"{features.shape=}")
        
        # Compression de la dimension fréquentielle en une seule dimension latente
        # Input: (batch, channels, freq_dim, time)
        # Output: (batch, latent_dim, 1, time)
        z = self.freq_projection(features)
        # print(f"Encoder {z.shape=}")

        # Retirer la dimension fréquentielle qui est maintenant de taille 1
        # Output: (batch, latent_dim, time)
        z = z.squeeze(2)
        
        return z, features
    
    def decode(self, z, features=None):
        # Ajouter une dimension fréquentielle de taille 1
        # Input: (batch, latent_dim, time)
        # Output: (batch, latent_dim, 1, time)
        z_expanded = z.unsqueeze(2)
        
        # Développer la dimension fréquentielle pour le décodeur
        # Output: (batch, channels, freq_dim, time)
        z_freq = self.freq_expansion(z_expanded)
        
        # Décodage standard
        x_hat = self.decoder(z_freq)
        
        return x_hat
    
    def forward(self, x):
        # Sauvegarder la dimension d'entrée pour vérification
        original_shape = x.shape
        
        # Encodage
        z, features = self.encode(x)
        
        # Attention temporelle optionnelle
        if self.use_attention:
            z_att = self.temporal_attention(z.unsqueeze(2)).squeeze(2)
            x_hat = self.decode(z_att)
        else:
            x_hat = self.decode(z)
            z_att = None
        
        # Forcer la dimension de sortie
        if x_hat.shape[2] != original_shape[2]:
            x_hat = nn.functional.interpolate(
                x_hat,
                size=(original_shape[2], original_shape[3]),
                mode='bilinear',
                align_corners=False
            )
        
        return x_hat, z, z_att, features

    # def forward(self, x):
    #     # Sauvegarder la dimension d'entrée pour vérification
    #     original_shape = x.shape
        
    #     # Encodage - compression en fréquence uniquement
    #     z = self.encoder(x)

    #     # Attention temporelle optionnelle
    #     if self.use_attention:
    #         z_att = self.temporal_attention(z)
    #         x_hat = self.decoder(z_att)
            
    #         # Forcer la dimension de sortie
    #         if x_hat.shape[2] != original_shape[2]:
    #             x_hat = nn.functional.interpolate(
    #                 x_hat,
    #                 size=(original_shape[2], original_shape[3]),
    #                 mode='bilinear',
    #                 align_corners=False
    #             )
                
    #         return x_hat, z_att, z    
    #     else:
    #         x_hat = self.decoder(z)
            
    #         # Forcer la dimension de sortie
    #         if x_hat.shape[2] != original_shape[2]:
    #             x_hat = nn.functional.interpolate(
    #                 x_hat,
    #                 size=(original_shape[2], original_shape[3]),
    #                 mode='bilinear',
    #                 align_corners=False
    #             )
                
    #         return x_hat, None, z


class DeconvBlock(nn.Module):
    """
    Bloc de déconvolution amélioré utilisant un upsampling suivi d'une convolution
    pour gérer les dimensions impaires.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 target_freq_dim=None, use_activation=True, act_slope=0.2):
        super(DeconvBlock, self).__init__()
        
        self.use_activation = use_activation
        self.target_freq_dim = target_freq_dim
        self.stride = stride
        
        # Utiliser un upsampling suivi d'une convolution au lieu de ConvTranspose2d
        # pour éviter les problèmes de dimensions
        self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
        
        # Convolution pour affiner les features après l'upsampling
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),  # Pas de downsampling ici
            padding=padding
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        
        if use_activation:
            self.act = nn.LeakyReLU(act_slope, inplace=True)
    
    def forward(self, x):
        # Upsampling initial
        x = self.upsample(x)
        
        # Convolution pour affiner
        x = self.conv(x)
        
        # Redimensionnement de la dimension fréquentielle si nécessaire
        if self.target_freq_dim is not None:
            batch_size, channels, height, width = x.size()
            if height != self.target_freq_dim:
                x = nn.functional.interpolate(
                    x, 
                    size=(self.target_freq_dim, width),
                    mode='bilinear',
                    align_corners=False
                )
        
        x = self.bn(x)
        
        if self.use_activation:
            x = self.act(x)
        
        return x


class ConvBlock(nn.Module):
    """
    Bloc de convolution standard pour l'encodeur
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, act_slope=0.2):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = nn.LeakyReLU(act_slope, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class TemporalAttention(nn.Module):
    """
    Module d'attention temporelle qui améliore les représentations latentes
    en tenant compte des relations temporelles.
    """
    def __init__(self, channels, hidden_dim=64):
        super(TemporalAttention, self).__init__()
        
        # Attention basée sur des convolutions temporelles dilatées
        self.attention_layers = nn.Sequential(
            # Convolution 1D temporelle avec dilatation croissante
            # pour capturer des dépendances à différentes échelles
            TemporalConvBlock(channels, hidden_dim, dilation=1),
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=2),
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=4),
            
            # Projection retour vers la dimension d'origine
            nn.Conv1d(hidden_dim, channels, kernel_size=1)
        )
        
        # Porte d'activation pour contrôler l'influence de l'attention
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [B, C, F, T]
        batch, channels, freq, time = x.size()
        
        # Reorganisation pour traitement temporel
        # [B, C, F, T] -> [B*F, C, T]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(-1, channels, time)
        
        # Calcul de l'attention temporelle
        attention = self.attention_layers(x_reshaped)
        gate = self.gate(x_reshaped)
        
        # Application de l'attention via mécanisme de porte (gating)
        enhanced = x_reshaped * gate + attention * (1 - gate)
        
        # Reconstruction de la forme d'origine
        # [B*F, C, T] -> [B, C, F, T]
        enhanced = enhanced.view(batch, freq, channels, time).permute(0, 2, 1, 3)
        
        return enhanced


class TemporalConvBlock(nn.Module):
    """
    Bloc de convolution temporelle avec dilatation pour capturer
    des dépendances à différentes échelles temporelles.
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(TemporalConvBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels, 
                out_channels,
                kernel_size=3, 
                padding=dilation,
                dilation=dilation
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Connection résiduelle si les dimensions correspondent
        self.use_residual = (in_channels == out_channels)
    
    def forward(self, x):
        out = self.conv_block(x)
        if self.use_residual:
            out = out + x
        return out




def test():

    audio_temporal_ae = AttentiveAE(hid_dim=[16, 32, 64, 32])
    audio_temporal_ae_att = AttentiveAE(hid_dim=[16, 32, 64, 32],
                                       use_attention=True)

    n_batch, n_chan, n_freq, n_frame = (64, 1, 513, 400)
    x = torch.randn((n_batch, n_chan, n_freq, n_frame))
    n_param_tae = sum(p.numel() for p in audio_temporal_ae.parameters() if p.requires_grad) * 1e-6
    n_param_tae_att = sum(p.numel() for p in audio_temporal_ae_att.parameters() if p.requires_grad) * 1e-6

    print("\nTesting AE w/o attention.....................")
    x_hat_tae, enc_out, z_tae = audio_temporal_ae(x)
    print(f"Number of parameters in temporal attention AE {n_param_tae:1.4f}M")
    print(f"{x.shape=}\n{x_hat_tae.shape=}\n{z_tae.shape=}")
    assert x.shape == x_hat_tae.shape, f"Wrong output shape with {type(audio_temporal_ae)}"
    print(".......................Done..................\n")

    print("\nTesting AE with attention....................")
    x_hat_tae, enc_out, z_tae = audio_temporal_ae_att(x)
    print(f"Number of parameters in temporal attention AE {n_param_tae_att:1.4f}M")
    print(f"{x.shape=}\n{x_hat_tae.shape=}\n{z_tae.shape=}")
    assert x.shape == x_hat_tae.shape, f"Wrong output shape with {type(audio_temporal_ae_att)}"
    print(".......................Done..................\n")

    print("\nTesting SpecAE...............................")
    spec_ae = SpecAE()
    x = torch.randn((64,1,32000))
    x_hat, x, z, z_enh = spec_ae(x)
    print(f"{x.shape=}\n{x_hat.shape=}\n{z.shape=}")
    assert x.shape == x_hat.shape, f"Wrong output shape with {type(spec_ae)}"
    print(".......................Done..................\n")


if __name__ == "__main__":
    test()