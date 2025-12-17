import torch
import torchaudio
import h5py
import tqdm
from glob import glob
from torchaudio.transforms import MFCC
import librosa # python -m pip install librosa
import custom_dsp_func
from custom_dsp_func.base_functions import loudness,spectral_energy_per_band,spectral_rms,hF_content_descriptor,dynamic_range,multi_level_acf
#from custom_dsp_func.essentia import dissonance
def sanitize_name(str_raw):
    
    str_cast = str(str_raw).lower()
    source=","
    target="-"
    remove="./()[]_"
    table = str.maketrans(source,target,remove)
    str_clean = str_cast.translate(table)
    
    return str_clean
class Concept():
    """
    Base class for a Concept that extract a given concept from a signal
    """
    def __init__(self,needStatistics = False):
        self.needStatistics = needStatistics
    def process(self,X):
        pass
    def get_name(self):
        return sanitize_name(type(self).__name__)
    def extract(self,input_folder,h5_file):
        """
        Extract the concept in the right hdf5 group.
        """
        concept_name = self.get_name()
        features = []
        flist = glob(f"{input_folder}/*.wav")
        for file in tqdm.tqdm(flist,desc=concept_name):
            audio,sr = torchaudio.load(file)
            features.append(self.__call__(audio))
        with h5py.File(h5_file,'w') as infile:
            
            dst = infile.require_group(concept_name)
            if self.needStatistics:
                mean,std = zip(*features)
                dst.create_dataset("value",dtype="f")
                dst.create_dataset("mean",data=mean)
                dst.create_dataset("std",data=std)
            else:
                
                dst.create_dataset("value",data=features)
                dst.create_dataset("mean",dtype="f")
                dst.create_dataset("std",dtype="f")
        return h5_file
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
    def get_name(self):
        return sanitize_name(f"{super().get_name()}_{self.n_mfcc}")
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
    
class Spectral_RollOff_Concept(Concept):
    """
    Compute the spectral roll-off using librosa
    Return two vectors of shape (Batch) for mean and std
    """
    def __init__(self,samplerate,rolloff_percentage = 0.9):
        super(Spectral_RollOff_Concept,self).__init__(needStatistics=True)
        self.samplerate = samplerate
        self.rolloff_percentage = rolloff_percentage
    def get_name(self):
        return sanitize_name(f"{super().get_name()}_{self.rolloff_percentage}")
    def process(self, X):
        return torch.from_numpy(librosa.feature.spectral_rolloff(y=X.numpy(),sr=self.samplerate,roll_percent = self.rolloff_percentage))

class Spectral_Centroid_Concept(Concept):
    """
    Compute the spectral centroid using librosa
    Return two vectors of shape (Batch) for mean and std
    """
    def __init__(self,samplerate):
        super(Spectral_Centroid_Concept,self).__init__(needStatistics=True)
        self.samplerate = samplerate
    def process(self, X):
        return torch.from_numpy(librosa.feature.spectral_centroid(y=X.numpy(),sr=self.samplerate))
    
class Loudness_Concept(Concept):
    """
    Compute the loudness using torchaudio
    Return one vector of shape (Batch):
    """
    def __init__(self,samplerate):
        super(Loudness_Concept,self).__init__(needStatistics=False)
        self.samplerate = samplerate
    
    def process(self,X):
        X = loudness(X,self.samplerate)
        return X

class MultiLevel_Acf_Concept(Concept):
    def __init__(self,samplerate,space=(0.1,1,100)):
        super(MultiLevel_Acf_Concept,self).__init__(needStatistics=True)
        self.samplerate = samplerate
        self.space = space
    def get_name(self):
        return sanitize_name(f"{super().get_name()}_{self.space}")
    def process(self,X):
        X,_ = multi_level_acf(X,self.samplerate,space=self.space)
        return X
class DynamicRange_Concept(Concept):
    """
    Compute the dynamic range using torchaudio
    Return one vector of shape (Batch):
    """
    def __init__(self,samplerate):
        super(DynamicRange_Concept,self).__init__(needStatistics=False)
        self.samplerate = samplerate
    
    def process(self,X):
        X = dynamic_range(X,self.samplerate)
        return X
    
class SpectralRMS_Concept(Concept):
    """
    Compute the spectral RMS using torchaudio
    Return one vector of shape (Batch):
    """
    def __init__(self,samplerate):
        super(SpectralRMS_Concept,self).__init__(needStatistics=False)
        self.samplerate = samplerate
    
    def process(self,X):
        X = dynamic_range(X,self.samplerate)
        return X

class HFContentDescriptor_Concept(Concept):
    """
    Compute the High frequency content descriptor using torchaudio
    Return two vectors (mean std) of shape (Batch):
    """
    def __init__(self,samplerate):
        super(HFContentDescriptor_Concept,self).__init__(needStatistics=True)
        self.samplerate = samplerate
    
    def process(self,X):
        X = hF_content_descriptor(X,self.samplerate)
        return X

class SpectralEnergyPerBand_Concept(Concept):
    """
    Compute the spectral energy per frequency band using torchaudio
    
    Return one vector of shape (Batch):
    """
    def __init__(self,samplerate,freqband):
        super(SpectralEnergyPerBand_Concept,self).__init__(needStatistics=False)
        self.samplerate = samplerate
        self.freqband = freqband
    def get_name(self):
        return sanitize_name(f"{super().get_name()}_{self.freqband}")
    def process(self,X):
        X = spectral_energy_per_band(X,self.samplerate,band=self.freqband)
        return X
    

#loudness
#dynamic range
#spectral rms
# HF content descriptor OK
# spectral energy per band OK
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
