import torch
import torchaudio
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
        X = X.numpy()
        spec_bw = librosa.feature.spectral_bandwidth(y = X,sr = self.samplerate)
        return torch.from_numpy(spec_bw)
    
class ZeroCrossingRate_Concept(Concept):
    """
    Compute the zero crossing rate using librosa
    Return two vectors for mean and std of shape (Batch):
    """
    def __init__(self,):
        super(ZeroCrossingRate_Concept,self).__init__(needStatistics=True)
        
    
    def process(self,X):
        X = X.numpy()
        zcr = librosa.feature.zero_crossing_rate(y = X)
        return torch.from_numpy(zcr)
    
class TemporalCentroid_Concept(Concept):
    """
    Compute the Temporal centroid
    Return one vector of shape (Batch):
    WIP
    """
    def __init__(self,samplerate=16000):
        super(TemporalCentroid_Concept,self).__init__(needStatistics=True)
        self.envelop_extraction_window = 0.01 #s
        self.samplerate = samplerate
        self.window = 100 #  ~0.1s
        self.threshold = 0.15

    def process(self,X):
        X_windowed = X.unfold(-1,int(self.envelop_extraction_window*self.samplerate),int(self.envelop_extraction_window*self.samplerate)) # B x nWin x lenWin
        global_envelop = X_windowed.abs().pow(2).sum(-1)
        envelop = global_envelop.unfold(-1,self.window,self.window)
        out_res = []
        for b_envelop in envelop: #process batch separately
            envMax = torch.max(b_envelop,dim=0)[0]

            overThreshEnvelop = b_envelop/envMax > self.threshold # Keep peak normalized frames with the most activation
            overThreshSupport = torch.stack([torch.arange(len(overThreshEnvelop[i])) for i in range(overThreshEnvelop.shape[0])])
            overThreshMean = (overThreshSupport*overThreshEnvelop).sum(dim=1) / overThreshEnvelop.sum(dim=1) # Balancing point in time (frame) weighted by energy
            out = overThreshMean / self.samplerate
            out_res.append(out)

        return torch.nan_to_num(torch.stack(out_res),nan=1e-7)
    
if __name__ == '__main__':
    #sample_wav_url = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
    X = torchaudio.load(torchaudio.utils.download_asset("tutorial-assets/steam-train-whistle-daniel_simon.wav"))[0]
    X_batched = torch.stack([X,X,X,X,X,X,X,X])[:,0,:] #mono
    print(f"{X_batched.shape=}")
    mfcc_extractor = MFCC_Concept(samplerate=44100)
    sbw_extractor = SpectralBandwith_Concept(samplerate=44100)
    zcr_extractor = ZeroCrossingRate_Concept()
    tempcent_extractor = TemporalCentroid_Concept(samplerate=44100)

    mfcc = mfcc_extractor(X_batched)
    print(f"{mfcc[0].shape=}, {mfcc[1].shape=}")

    sbw = sbw_extractor(X_batched)
    print(f"{sbw[0].shape=}, {sbw[1].shape=}")
    
    zcr = zcr_extractor(X_batched)
    print(f"{zcr[0].shape=}, {zcr[1].shape=}")

    tempcent = tempcent_extractor(X_batched)
    print(f"{tempcent[0].shape=}, {tempcent[1].shape=}")
