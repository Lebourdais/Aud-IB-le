import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as tr
from audIBle.nn.Tcn import TCN
class Encoder(nn.Module):
    def __init__(self,dim = 512, mel=80,sample_rate=44100):
        super(Encoder,self).__init__()
        self.mel = tr.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=mel
            )
        self.tcn = TCN(in_chan=80,n_src=1,out_chan=dim,n_repeats=3,n_blocks=5)
        
        # an affine operation: y = Wx + b
    def forward(self,X):
        X_mel = self.mel(X).squeeze()
        c1 = self.tcn(X_mel)
        return c1
        

if __name__=='__main__':
    dummy = torch.rand((8,1,44100*5))
    model = Encoder(dim=512)
    model(dummy)