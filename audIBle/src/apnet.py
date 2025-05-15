import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from audIBle.nn.apnet_layers import PrototypeLayer
from audIBle.nn.apnet_layers import WeightedSum
import math

def normalize_wav(wav):
    energy = torch.sum(wav ** 2, dim=-1, keepdim=True)
    energy = torch.clamp(energy, min=1e-6)  # Avoid division by zero
    wav = wav / torch.sqrt(energy)
    return wav

class APNetEncoder(nn.Module):
    def __init__(self, n_filters, filter_size, pool_size, dilation_rate=(1, 1), use_batch_norm=False):
        super(APNetEncoder, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.pool_size = pool_size
        self.dilation_rate = dilation_rate

        # Define the layers
        self.conv1 = nn.Conv2d(1, n_filters[0], kernel_size=filter_size, padding=filter_size // 2, dilation=dilation_rate[0])
        self.bn1 = nn.BatchNorm2d(n_filters[0]) if use_batch_norm else nn.Identity()
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(n_filters[0], n_filters[1], kernel_size=filter_size, padding=filter_size // 2, dilation=dilation_rate[1])
        self.bn2 = nn.BatchNorm2d(n_filters[1]) if use_batch_norm else nn.Identity()
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(n_filters[1], n_filters[2], kernel_size=filter_size, padding=filter_size // 2)
        self.bn3 = nn.BatchNorm2d(n_filters[2]) if use_batch_norm else nn.Identity()
        self.relu3 = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.upsample = nn.Upsample(scale_factor=pool_size, mode='nearest')

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        orig = x
        x = self.pool(x)
        y_up = self.upsample(x)

        # Mask1: where original is greater than upsampled
        mask1 = (orig >= y_up).float()

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        orig = x
        x = self.pool(x)
        y_up = self.upsample(x)

        # Mask2: where original is greater than upsampled
        mask2 = (orig >= y_up).float()

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x, mask1, mask2



class APNetDecoder(nn.Module):
    def __init__(self, n_filters, filter_size, pool_size, use_batch_norm=False, final_activation='tanh', n_filters_out=1):
        super(APNetDecoder, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.pool_size = pool_size
        self.final_activation = final_activation
        self.n_filters_out = n_filters_out

        # Use padding that mimics 'same' padding
        self.deconv1 = nn.ConvTranspose2d(n_filters[2], n_filters[1], kernel_size=filter_size, stride=1, padding=filter_size//2, output_padding=0)
        self.bn1 = nn.BatchNorm2d(n_filters[1]) if use_batch_norm else nn.Identity()
        self.relu1 = nn.LeakyReLU()

        self.deconv2 = nn.ConvTranspose2d(n_filters[1], n_filters[0], kernel_size=filter_size, stride=1, padding=filter_size//2, output_padding=0)
        self.bn2 = nn.BatchNorm2d(n_filters[0]) if use_batch_norm else nn.Identity()
        self.relu2 = nn.LeakyReLU()

        self.deconv3 = nn.ConvTranspose2d(n_filters[0], 1, kernel_size=filter_size, stride=1, padding=filter_size//2, output_padding=0)
        self.bn3 = nn.BatchNorm2d(n_filters_out) if use_batch_norm else nn.Identity()

        self.upsample = nn.Upsample(scale_factor=pool_size, mode='nearest')

        self.final_act = nn.Tanh() if final_activation == 'tanh' else nn.Identity()

    def forward(self, x, mask1, mask2):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.upsample(x)
        x = mask2 * x
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.upsample(x)
        x = mask1 * x
        x = self.relu2(x)

        x = self.deconv3(x)
        if self.n_filters_out == 1:
            x = x.squeeze(1)  # Remove channel dim if single-channel

        x = self.bn3(x)
        x = self.final_act(x)

        return x


class APNet(nn.Module):
    def __init__(self, 
                 n_classes=10,
                 seg_len=5, # duration in seconds, necessarily fixed in this model :(
                 n_filters=[64, 128, 256], 
                 filter_size=3, 
                 pool_size=2, 
                 dilation_rate=(1, 1), 
                 use_batch_norm=False, 
                 final_activation='tanh',
                 n_prototypes=10,
                 distance='euclidean',
                 use_weighted_sum=True,
                 mel_spec_param={"sample_rate":44100, "n_fft":4096, "hop_length":1024, "n_mels":256, "normalize": True},):
        
        super(APNet, self).__init__()
        nfreq = mel_spec_param["n_mels"]
        duration_sample = seg_len * mel_spec_param["sample_rate"]
        ntime = math.ceil(duration_sample // mel_spec_param["hop_length"])+1
        pad_size = 4 - (ntime % 4)
        self.spec = torchaudio.transforms.MelSpectrogram(**mel_spec_param)
        self.encoder = APNetEncoder(n_filters, filter_size, pool_size, dilation_rate, use_batch_norm)
        self.decoder = APNetDecoder(n_filters, filter_size, pool_size, use_batch_norm, final_activation)
        self.proto_layer = PrototypeLayer(n_prototypes=n_prototypes, n_chan_latent=n_filters[-1], n_freq_latent=nfreq//4, n_frames_latent=(ntime+pad_size)//4, distance=distance, use_weighted_sum=use_weighted_sum)
        self.weighted_sum = WeightedSum(D3=False, n_freq_latent=nfreq//4, n_prototypes=n_prototypes)
        self.classif = nn.Linear(n_prototypes, n_classes, bias=False)
        self.activation = nn.Softmax(dim=1)
        self.n_classes = n_classes
        self.pad_size = pad_size

    def forward(self, wav, return_all=False, return_mask = False):
        wav = normalize_wav(wav)
        spec_feat = self.spec(wav)
        spec_feat = torch.log(1 + spec_feat) 
        if spec_feat.shape[-1] % 4 != 0:
            # Pad the last dimension to be divisible by 4
            spec_feat = F.pad(spec_feat, (0, self.pad_size), mode='constant', value=0)

        # print("Shape after MelSpectrogram:", spec_feat.shape)

        z, mask1, mask2 = self.encoder(spec_feat)
        # print("Shape after encoder:", z.shape)
        #normalize Z
        z = F.normalize(z, p=2)

        x_hat = self.decoder(z, mask1, mask2)
        # print("Shape after decoder:", x_hat.shape)
    
        # distance between z and prototypes
        sim = self.proto_layer(z)
        # print("Distance full:", sim)
        # print("Shape after prototype layer:", sim.shape)
        
        d = torch.exp(-sim)
        # print(f"distance: {d}")
        # print("Shape after applying exp to distances:", d.shape)

        # weighted sum of prototypes
        pred = self.weighted_sum(d) 
        # print("Prediction weighted sum:", pred)
        # print("Shape after weighted sum:", pred.shape)

        # classification
        pred = self.classif(pred)
        # print("Prediction:", pred)
        # print("Shape after classification:", pred.shape)
        if return_all:
            if return_mask:
                return pred, x_hat.squeeze(1), sim, spec_feat.squeeze(1), z, mask1, mask2
            return pred, x_hat.squeeze(1), sim, spec_feat.squeeze(1), z
        return pred, x_hat.squeeze(1), sim, spec_feat.squeeze(1)

    def get_prototypes(self):
        proto = self.proto_layer.kernel
        # Normalize the prototypes
        proto = F.normalize(proto, p=2)
        return proto