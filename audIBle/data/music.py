import pandas
import torchaudio
import torchaudio.transforms as T
import os
from torch.utils.data import Dataset
from datasets import load_dataset,Audio


class GTZAN(Dataset):
    
    def __init__(self,csv_file,root_dir,mode="train",prop=0.8,seed=42,target_samplerate=16000,overwrite=False):
        super().__init__()
        
        self.samplerate = 22050
        self.target_samplerate = target_samplerate
        
        if not os.path.exists(f"{root_dir}/train.csv") or overwrite:
            data_full = pandas.read_csv(f"{root_dir}{csv_file}")
            
            data_full = data_full[~data_full['filename'].str.contains("jazz.00054")]
            
            data_train = data_full.sample(frac=prop,random_state=seed)
            data_test = data_full.loc[data_full.index.difference(data_train.index)]
            
            data_valid = data_train.sample(frac=0.1,random_state=seed)
            data_train = data_train.loc[data_train.index.difference(data_valid.index)]
            data_train.to_csv(f"{root_dir}/train.csv")
            data_valid.to_csv(f"{root_dir}/valid.csv")
            data_test.to_csv(f"{root_dir}/test.csv")
        
        if mode == "train":
            self.data = pandas.read_csv(f"{root_dir}/train.csv")
        elif mode == "valid":
            self.data = pandas.read_csv(f"{root_dir}/valid.csv")
        else:
            self.data = pandas.read_csv(f"{root_dir}/test.csv")

        self.genre_list = os.listdir(f"{root_dir}/genres_original/")
        self.genre_list.sort()
        
        self.label_encode = {}
        for ii,genre in enumerate(self.genre_list):
            self.label_encode[genre] = ii
            
        
        num_win = 10
        length = self.data.iloc[0]["length"]
        
        self.windows = [{"idx_start":t*length, "idx_stop":(t+1)*length} for t in range(num_win)]
        self.root_dir = root_dir+"genres_original/"
        self.seg_length = length
        self.resampler = T.Resample(self.samplerate, self.target_samplerate)
        
    def __getitem__(self,idx):
        seg_info = self.data.iloc[idx]
        label = seg_info["label"]
        name_split = seg_info["filename"].split(".")
        win_idx = int(name_split[2])
        
        uri = f"{self.root_dir}{label}/{label}.{name_split[1]}.wav"
        audio,_ = torchaudio.load(uri,
                                frame_offset=self.windows[win_idx]["idx_start"],
                                num_frames=self.seg_length)
        audio = self.resampler(audio)
        
        return audio, self.label_encode[label]
        
    def __len__(self):
        return len(self.data)
    
    def get_classes(self):
        return self.genre_list