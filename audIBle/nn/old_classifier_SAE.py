import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from audIBle.nn.sae import SAE
import torchaudio

def normalize_wav(wav):
    energy = torch.sum(wav ** 2, dim=-1, keepdim=True)
    energy = torch.clamp(energy, min=1e-6)  # Avoid division by zero
    wav = wav / torch.sqrt(energy)
    return wav

class SparseClassifier(nn.Module):

    def __init__(self,
                 n_classes,
                 audio_ae_type='temporal',
                 feat_type="stft",
                 sae_kw = {"sae_dim": 256,"sparsity": 0.05, "method": "top-k"},
                 spec_kw = {"win_length": 1024, "hop_length": 256, "n_fft": 1024},
                 audio_ae_kw={"input_channels":1, "hid_dim": [16, 32, 64, 128],"use_attention":False, "activation_slope": 0.2, "attention_dim": 64}):
        super(SparseClassifier,self).__init__()
        sae_kw["input_dim"] = audio_ae_kw["hid_dim"][-1]
        audio_ae_kw["input_freq_dim"] = spec_kw["n_fft"]//2+1


        if feat_type == "stft":
            self.spec = torchaudio.transforms.Spectrogram(**spec_kw)
        elif feat_type == "mel":
            self.spec = torchaudio.transforms.MelSpectrogram(**spec_kw)

        if audio_ae_type == "vanilla":
            self.audio_ae = AudioAE(**audio_ae_kw)
        elif audio_ae_type == "temporal":
            self.audio_ae = AudioAttAE(**audio_ae_kw)
        
        self.sae = SAE(**sae_kw)
        self.classif_head = nn.Linear(sae_kw["sae_dim"],n_classes,bias=False)

    def forward(self, wav):
        """
        Forward audio through the model. 
        Returns: 
            `y_hat`: classification logits
            `x_hat`: decoded audio
            `enc_audio_hat`: SAE output in the latent space
            `enc_audio`: SAE input (for MSE loss)
        """
        wav = normalize_wav(wav)
        x = self.spec(wav)
        x = torch.log(1 + x) 
        
        # audo-encode the audio
        x_hat, A_enh, A = self.audio_ae(x)
        
        # sparse autoencoder applied to the latent representation of the input
        A_hat, Z = self.sae(A)
        
        # classify from the sparse latent representation
        y_hat = self.classif_head(Z)

        return y_hat, x_hat, A_hat, A, Z

class ConvBlock(nn.Module):
    def __init__(self,
                 input_channels, 
                 output_channels, 
                 kernel_size, 
                 stride, 
                 padding, 
                 act_slope=0.0):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(output_channels)
        self.act = nn.LeakyReLU(act_slope)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self,
                 scale_factor,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 padding,
                 use_activation=True,
                 act_slope=0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(act_slope)
        self.use_activation = use_activation

    def forward(self,x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        if self.use_activation:
            x = self.act(x)
        return x



class AudioAE(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 hidden_channels=[16, 32, 64, 128], 
                 stride=2,
                 kernel_size=3,
                 activation_slope=0.0):
        super(AudioAE, self).__init__()
        
        # Encodeur
        enc_layers = OrderedDict()
        in_chan=input_channels
        for i, h in enumerate(hidden_channels):
            enc_layers[f"block_{i}"] = ConvBlock(input_channels=in_chan, 
                                                 output_channels=h, 
                                                 kernel_size=kernel_size, 
                                                 stride=stride, 
                                                 padding=1,
                                                 act_slope=activation_slope)
            in_chan = h

        self.encoder = nn.Sequential(enc_layers)
        
        dec_layers = OrderedDict()
        n_block = len(hidden_channels)
        for i in range(n_block):
            in_chan = hidden_channels.pop()
            dec_layers[f"block_{i}"] = DeconvBlock(scale_factor=stride, 
                                                   in_channels=in_chan, 
                                                   out_channels=in_chan//2 if len(hidden_channels)!= 0 else input_channels,
                                                   kernel_size=kernel_size,
                                                   padding=1,
                                                   use_activation= i != (n_block-1),
                                                   act_slope=activation_slope)


        # Décodeur avec Upsample + Conv pour éviter les artefacts en damier
        self.decoder = nn.Sequential(dec_layers)
        
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z

class AudioAttAE(nn.Module):
    """
    Auto-encodeur qui compresse uniquement la dimension fréquentielle tout en
    préservant intégralement la dimension temporelle.
    
    Structure du spectrogramme: (batch, channels, fréquence, temps)
    """
    def __init__(self, 
                 input_channels=1, 
                 input_freq_dim=128, 
                 hid_dim=[16, 32, 64, 128],
                 use_attention=False,
                 activation_slope=0.2,
                 attention_dim=64):
        super(AudioAttAE, self).__init__()
        
        self.input_freq_dim = input_freq_dim
        print(hid_dim)
        self.latent_dim = hid_dim[-1]
        kernel_size = (5,3)
        padding = (2,1)
        stride = (2,1)
        
        n_block = len(hid_dim)
        enc_layers = OrderedDict()
        for ii in range(n_block):
            enc_layers[f"block_{ii}"] = ConvBlock(input_channels=input_channels if ii == 0 else hid_dim[ii-1], 
                                                 output_channels=hid_dim[ii], 
                                                 kernel_size=kernel_size, 
                                                 stride=stride, 
                                                 padding=padding,
                                                 act_slope=activation_slope)
            
        self.encoder = nn.Sequential(enc_layers)
        
        if use_attention:
            self.temporal_attention = TemporalAttention(self.latent_dim, attention_dim)
        
        dec_layers = OrderedDict()
        rev_hid_dim = list(reversed(hid_dim))
        for jj, h in enumerate(rev_hid_dim):
            if jj == n_block-1:
                out = input_channels
            else:
                out = rev_hid_dim[jj+1]
            dec_layers[f"block_{jj}"] = DeconvBlock(scale_factor=stride, 
                                                   in_channels=h, 
                                                   out_channels=out,
                                                   kernel_size=kernel_size,
                                                   padding=padding,
                                                   use_activation= jj != (n_block-1),
                                                   act_slope=activation_slope)

        self.decoder = nn.Sequential(dec_layers)
        self.use_attention = use_attention
    
    def forward(self, x):
        # Encodage - compression en fréquence uniquement
        z = self.encoder(x)
        
        # Attention temporelle optionnelle
        if self.use_attention:
            z_att = self.temporal_attention(z)
            x_hat = self.decoder(z_att)
            return x_hat, z_att, z    
        else:
            x_hat = self.decoder(z)
            return x_hat, None, z


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

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        batch_size = x.shape[0]
        if self.shape[0] == -1:  # Cas pour bidirectional LSTM
            # [B, C, H, W] -> [B*H, C, W] pour traiter chaque ligne de fréquence indépendamment
            return x.permute(0, 2, 1, 3).contiguous().view(batch_size * x.shape[2], self.shape[1], x.shape[3])
        else:  # Cas pour retour à la forme d'origine
            # [B*H, 2*hidden, W] -> [B, C, H, W]
            h_dim = x.shape[0] // batch_size
            return x.view(batch_size, h_dim, self.shape[0], x.shape[2]).permute(0, 2, 1, 3)
        

def test():

    audio_ae = AudioAE()
    audio_temporal_ae = AudioAttAE(hid_dim=[16, 32, 64, 32])
    audio_temporal_ae_att = AudioAttAE(hid_dim=[16, 32, 64, 32],
                                       use_attention=True)
    print(audio_temporal_ae_att)


    n_batch, n_chan, n_freq, n_frame = (64, 1, 512, 400)
    x = torch.randn((n_batch, n_chan, n_freq, n_frame))
    n_param_ae = sum(p.numel() for p in audio_ae.parameters() if p.requires_grad) * 1e-6
    n_param_tae = sum(p.numel() for p in audio_temporal_ae.parameters() if p.requires_grad) * 1e-6
    n_param_tae_att = sum(p.numel() for p in audio_temporal_ae_att.parameters() if p.requires_grad) * 1e-6
    
    x_hat_ae, z_ae = audio_ae(x)
    print(f"Number of parameters in vanilla AE {n_param_ae:1.4f}M")
    print(f"{x.shape=} | {x_hat_ae.shape=} | {z_ae.shape=}")
    
    assert x.shape == x_hat_ae.shape, f"Wrong output shape with {type(audio_ae)}"

    x_hat_tae, enc_out, z_tae = audio_temporal_ae(x)
    print(f"Number of parameters in temporal attention AE {n_param_tae:1.4f}M")
    print(f"{x.shape=} | {x_hat_tae.shape=} | {z_tae.shape=}")
    assert x.shape == x_hat_tae.shape, f"Wrong output shape with {type(audio_temporal_ae)}"

    x_hat_tae, enc_out, z_tae = audio_temporal_ae_att(x)
    print(f"Number of parameters in temporal attention AE {n_param_tae_att:1.4f}M")
    print(f"{x.shape=} | {x_hat_tae.shape=} | {z_tae.shape=}")
    assert x.shape == x_hat_tae.shape, f"Wrong output shape with {type(audio_temporal_ae_att)}"


    print("Test audio classifier ...................")

    sparse_classif = SparseClassifier()

    print("Test done!")

if __name__ == "__main__":
    test()