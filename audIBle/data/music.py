import pandas
import torchaudio
import torchaudio.transforms as T
import os
from torch.utils.data import Dataset

import torch
import glob
from pathlib import Path
from typing import Tuple, Union
import random

import pandas as pd


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
    

class VocalSet(Dataset):
    def __init__(self, root: Union[str, Path],
                 split: str,
                 seed: int = None,
                 ratio: float = None,
                 duration: float = 5.0,
                 target_sr: int = 16000) -> None:
        """
        Original code from Russell Izadi 2023 (https://github.com/Russell-Izadi-Bose/svae/blob/main/vocalset.py)
        Modified by Théo Mariotte 2025
        """
        if split == "test":
            root = os.path.join(root, "test")
        else:
            root = os.path.join(root, "train")

        list_paths = glob.glob(os.path.join(
            root, '**/*.wav'), recursive=True)

        if split != "test":
            list_paths_train, list_paths_valid = self.random_split(
            list_paths, seed=seed, ratio=ratio)
            if split == 'train':
                list_paths = list_paths_train
            elif split == 'valid':
                list_paths = list_paths_valid

        self.list_paths = list_paths
        self.data = self.get_data(list_paths)

        # clean dataset
        self._preprocess_data()

        self.target_sr = target_sr
        # length of the training segment
        self.length = int(target_sr * duration) if duration is not None else None
        self.orig_sr = 44100
        self.resample = torchaudio.transforms.Resample(orig_freq=self.orig_sr, new_freq=self.target_sr)
        self.split = split
        self.techniques = ['belt', 'breathy', 'inhaled', 'lip_trill', 'spoken', 'straight', 'trill', 'trillo', 'vibrato', 'vocal_fry']
        self.techniques_map = {technique: idx for idx, technique in enumerate(self.techniques)}
        

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, set, str]:
        if i >= len(self):
            raise IndexError
        
        path_wav = self.data["path"].iloc[i]
        label = self.techniques_map[self.data["technique"].iloc[i]]
        # path_wav, label_full = self.data[i]

        audio, sr = torchaudio.load(path_wav)

        if self.orig_sr != self.target_sr:
            audio = self.resample(audio)

        audio_len = audio.shape[-1]
        
        if self.length is not None:
            if audio_len < self.length:
                pad_len = self.length - audio_len
                audio = torch.nn.functional.pad(audio, (0, pad_len))
            if audio_len > self.length:
                if self.split == "train":
                    start = random.randint(0, audio_len - self.length)
                else:
                    start = 0
                audio = audio[:, start:start + self.length]

        return audio, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data(list_path_wavs):

        data = []
        for path in list_path_wavs:
            singer = path.split("/")[-4]
            context = path.split("/")[-3]
            technique = path.split("/")[-2]
            vowel = path.split("/")[-1].split(".")[0].split("_")[-1]

            label = {
                'singer': singer,
                'context': context,
                'technique': technique,
                'vowel': vowel}

            data.append((path, label))
            
        return data
    
    def _preprocess_data(self):
        """
        Method to filter unused audio segments
        """
        # keep only 10 singing techniques as in the original paper
        df = pd.DataFrame([
            {'path': path, **label}
            for path, label in self.data])
        
        # rename vibrado (typo in the labels)
        df.loc[df["technique"] == "vibrado"] = "vibrato"

        # keep only the 10 techniques of interest
        df_tech = df.loc[df['technique'].isin(['belt', 'breathy', 'inhaled', 'lip_trill','spoken', 'straight', 'trill','trillo', 'vibrato', 'vocal_fry'])]

        # keep only the vowels and spoken signals (other vowels represent the uncut audio)
        df_final = df_tech.loc[df_tech["vowel"].isin(['e', 'o', 'i', 'a', 'u', 'c','f','spoken'])]

        self.data = df_final


    @staticmethod
    def random_split(list_, seed=0, ratio=0.8):
        random.seed(seed)
        random.shuffle(list_)
        n_samples = int(len(list_) * ratio)
        return list_[:n_samples], list_[n_samples:]