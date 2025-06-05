import torch
from torchaudio.transforms import MFCC
import librosa

class Concept():
    """
    Base class for a Concept that extract a given concept from a signal
    """
    def __init__(self,needStatistics = False):
        self.needStatistics = needStatistics
    def process(self,X):
        pass

    def __call__(self, X):
        """Treat signal X"""
        out = self.process(X)
        if self.needStatistics:
            out = (torch.mean(out,dim=-1),torch.std(out,dim=-1))
        else:
            out = out
        return out
    
class MFCC_Concept(Concept):
    """
    Return two vectors: 
    - mean : Batch,n_mfcc
    - std : Batch,n_mfcc
    """
    def __init__(self,n_mfcc=12,samplerate = 16000):
        super(MFCC_Concept, self).__init__(needStatistics=True)
        self.n_mfcc = n_mfcc
        self.transform = MFCC(sample_rate=samplerate,n_mfcc=n_mfcc)

    def process(self,X):
        return self.transform(X)
    
class SpectralBandwith_Concept(Concept):
    """
    Compute the spectral bandwith using librosa
    Return two vectors for mean and std of shape (Batch):
    """
    def __init__(self, samplerate=16000):
        super(SpectralBandwith_Concept,self).__init__(needStatistics=True)
        self.samplerate = samplerate
    
    def process(self,X):
        spec_bw = librosa.feature.spectral_bandwidth(y = X,sr = self.samplerate)
        return spec_bw
    
class ZeroCrossingRate_Concept(Concept):
    """
    Compute the zero crossing rate using librosa
    Return two vectors for mean and std of shape (Batch):
    """
    def __init__(self,):
        super(ZeroCrossingRate_Concept,self).__init__(needStatistics=True)
        
    
    def process(self,X):
        zcr = librosa.feature.zero_crossing_rate(y = X)
        return zcr
    
class TemporalCentroid_Concept(Concept):
    """
    Compute the Temporal centroid
    Return one vector of shape (Batch):
    WIP
    """
    def __init__(self,):
        super(TemporalCentroid_Concept,self).__init__(needStatistics=False)
        self.window = 512
    
    def process(self,X):
        X_windowed = X.unfold(self.window,self.window,dim=-1)

        
        return 0