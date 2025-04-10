import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as tr
from audIBle.nn.Tcn import TCN
class Classifier(nn.Module):
    def __init__(self,dim = 512, n_class=80,sample_rate=44100):
        super(Classifier,self).__init__()
       
        self.tcn = TCN(in_chan=dim*2,n_src=1,out_chan=n_class,n_repeats=2,n_blocks=3)
        
        # an affine operation: y = Wx + b
    def forward(self,X):
        Xcat = torch.concatenate((X.mean(-1),X.std(-1)),dim=-1).unsqueeze(-1)
        
        c1 = self.tcn(Xcat)
        return c1
        

if __name__=='__main__':
    dummy = torch.rand((8, 480, 431))
    model = Classifier(dim=480,n_class=50)
    model(dummy)