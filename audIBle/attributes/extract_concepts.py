import numpy as np
import torchaudio
import glob
import os
import h5py as h5
import tqdm
from concepts import MFCC_Concept,Spectral_Centroid_Concept,Spectral_RollOff_Concept,ZeroCrossingRate_Concept,MultiLevel_Acf_Concept,SpectralBandwith_Concept,TemporalCentroid_Concept,Loudness_Concept,SpectralEnergyPerBand_Concept,SpectralRMS_Concept,DynamicRange_Concept,HFContentDescriptor_Concept

def pad_lists(list_array):
    mapping = np.zeros([len(list_array), max([x.shape[-1]for x in list_array])])
    for i, sub in enumerate(list_array):
        mapping[i][0:sub.shape[-1]] = sub
    return mapping

def merge_db(dict_db):
    """
    To merge multiple database into a single endpoint
    the dict should be of shape
    {root_name:file_path}

    """
    fname = "BigDB.hdf5"
    with h5.File(fname,"a") as infile:
        # Create dataset
        dst = infile.require_group("main")
        for db in dict_db:
            dst[db] = h5.ExternalLink(dict_db[db],f"/{db}")


def extract_db(file):
    """
    Extract the ESC-50 db
    """
    overwrite = False
    input_folder = "/mnt/data/ESC-50-master/audio"
    flist = glob.glob(f"{input_folder}/*.wav")
    _,sr = torchaudio.load(flist[0])
    concepts = [MFCC_Concept(samplerate=sr),
                SpectralBandwith_Concept(samplerate=sr),
                ZeroCrossingRate_Concept(),
                TemporalCentroid_Concept(samplerate=sr),
                Spectral_RollOff_Concept(samplerate=sr),
                Spectral_Centroid_Concept(samplerate=sr),
                Loudness_Concept(samplerate=sr),
                DynamicRange_Concept(samplerate=sr),
                SpectralRMS_Concept(samplerate=sr),
                HFContentDescriptor_Concept(samplerate=sr),
                SpectralEnergyPerBand_Concept(samplerate=sr,freqband=(20,150)),
                MultiLevel_Acf_Concept(samplerate=sr) # A bit long
                ]
    with h5.File(file,"a") as infile:
        # Create dataset
        dst = infile.require_group("ESC50")
        if "filename" not in dst:
            dst["filename"] = np.asarray([os.path.basename(x) for x in flist],dtype='T')

        for c in tqdm.tqdm(concepts):
            if c.get_name() in dst and not overwrite:
                
                continue            
            ext_file = c.extract(input_folder,f"{c.get_name()}.hdf5")
            dst[c.get_name()] = h5.ExternalLink(ext_file,f"/{c.get_name()}")
            
            
                
            
            


if __name__ == '__main__':
    extract_db("Test.hdf5")