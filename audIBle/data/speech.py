import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset,Audio
import os
import torch
import glob
from typing import List, Tuple, Dict
import scipy
import numpy as np

class CommonVoiceDataset(Dataset):
    def __init__(self,mode="train",samplerate=16000,length=2.0,label="gender"):
        super().__init__()
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split=mode)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=samplerate))
        dataset = dataset.remove_columns(["sentence","up_votes","down_votes","age","accent","locale","segment"])
        print("Cleaning...",end="")
        dataset = dataset.filter(lambda example: example,input_columns=label)
        print("..",end="")
        dataset = dataset.map(self.to_index,input_columns=label)
        dataset.set_format(type='torch', columns=['audio', 'gender'])
        print("..OK")
        self.samplerate = samplerate
        self.set = dataset
        self.length=length
        self.sample_length=int(length*samplerate)
        
        
    def to_index(self,example):
        example = 1 if example=='female' else 0
        dicti={"gender":example}
        return dicti
    
    def __getitem__(self,idx):
        sample = self.set.__getitem__(idx)
        label = sample['gender']
        audio = sample['audio']['array']
        
        
        audio = F.pad(audio[:self.sample_length],(0,max(0,self.sample_length-audio.shape[0])), "constant", 0)
        return audio,label
    def __len__(self):
        return len(self.set)



import os
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import scipy.io.wavfile
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

class TimitDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", subsplit: str = None, 
                 samplerate: int = 16000, length: float = 0.5, 
                 train_ratio: float = 0.8, random_seed: int = 42, overwrite: bool = False):
        """
        Custom TIMIT Dataset Loader with train/validation splitting
        
        Args:
            data_dir: Path to TIMIT root directory
            split: "train" or "test"
            subsplit: "train" or "valid" (only used when split="train")
            samplerate: Target sampling rate
            length: Length of audio segments in seconds
            train_ratio: Ratio of training data (rest goes to validation)
            overwrite: Whether to overwrite existing split files
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split.lower()
        self.subsplit = subsplit.lower() if subsplit else None
        self.samplerate = samplerate
        self.length = length
        self.sample_length = int(length * samplerate)
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.overwrite = overwrite
        
        # Phoneme mapping (same as your original)
        self.mapping_phone = {
            'iy': 0, 'ih': 1, 'ix': 1, 'eh': 2, 'ae': 3, 'ax': 4, 'ah': 4, 'ax-h': 4,
            'uw': 5, 'ux': 5, 'uh': 6, 'ao': 7, 'aa': 7, 'ey': 8, 'ay': 9, 'oy': 10,
            'aw': 11, 'ow': 12, 'er': 13, 'axr': 13, 'l': 14, 'el': 14, 'r': 15,
            'w': 16, 'y': 17, 'm': 18, 'em': 18, 'n': 19, 'en': 19, 'nx': 19,
            'ng': 20, 'eng': 20, 'v': 21, 'f': 22, 'dh': 23, 'th': 24, 'z': 25,
            's': 26, 'zh': 27, 'sh': 27, 'jh': 28, 'ch': 29, 'b': 30, 'p': 31,
            'd': 32, 'dz': 33, 't': 34, 'g': 35, 'k': 36, 'hh': 37, 'hv': 37,
        }
        
        self.phone_list = ['iy', 'ih', 'eh', 'ae', 'ax', 'uw', 'uh', 'ao', 'ey', 'ay', 
                          'oy', 'aw', 'ow', 'er', 'l', 'r', 'w', 'y', 'm', 'n', 'ng', 
                          'v', 'f', 'dh', 'th', 'z', 's', 'zh', 'jh', 'ch', 'b', 'p', 
                          'd', 'dz', 't', 'g', 'k', 'hh', 'sil']
        
        # Load and process data
        self.phone_segments = []
        self._load_data()
        
    def _get_split_file_paths(self):
        """Get paths for train/valid split files"""
        train_split_file = os.path.join(self.data_dir, f"{self.split}_train_files.json")
        valid_split_file = os.path.join(self.data_dir, f"{self.split}_valid_files.json")
        return train_split_file, valid_split_file
    
    def _create_train_valid_split(self, wav_files: List[str]) -> Tuple[List[str], List[str]]:
        """
        Create or load train/validation split
        
        Args:
            wav_files: List of all wav file paths
            
        Returns:
            Tuple of (train_files, valid_files)
        """
        train_split_file, valid_split_file = self._get_split_file_paths()
        
        # Check if split files exist and overwrite is False
        if not self.overwrite and os.path.exists(train_split_file) and os.path.exists(valid_split_file):
            print(f"Loading existing train/valid split from {train_split_file} and {valid_split_file}")
            try:
                with open(train_split_file, 'r') as f:
                    train_files = json.load(f)
                with open(valid_split_file, 'r') as f:
                    valid_files = json.load(f)
                
                # Verify that all files still exist
                train_files = [f for f in train_files if os.path.exists(f)]
                valid_files = [f for f in valid_files if os.path.exists(f)]
                
                print(f"Loaded {len(train_files)} train files and {len(valid_files)} valid files")
                return train_files, valid_files
                
            except Exception as e:
                print(f"Error loading existing split files: {e}")
                print("Creating new split...")
        
        # Create new split
        print(f"Creating new train/valid split with ratio {self.train_ratio}")
        
        # Sort files for reproducibility
        wav_files = sorted(wav_files)
        
        # Split files
        np.random.seed(self.random_seed)
        train_files, valid_files = train_test_split(
            wav_files, 
            train_size=self.train_ratio, 
            random_state=self.random_seed,
            shuffle=True
        )
        
        # Save split files
        try:
            with open(train_split_file, 'w') as f:
                json.dump(train_files, f, indent=2)
            with open(valid_split_file, 'w') as f:
                json.dump(valid_files, f, indent=2)
            print(f"Saved train/valid split to {train_split_file} and {valid_split_file}")
        except Exception as e:
            print(f"Warning: Could not save split files: {e}")
        
        print(f"Created split: {len(train_files)} train files, {len(valid_files)} valid files")
        return train_files, valid_files
    
    def _load_data(self):
        """Load all audio files and phoneme annotations"""
        split_dir = os.path.join(self.data_dir, self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        print(f"Loading {self.split} data from {split_dir}")
        
        # Find all .wav files
        all_wav_files = glob.glob(os.path.join(split_dir, "**", "*.wav"), recursive=True)
        print(f"Found {len(all_wav_files)} total audio files")
        
        # Handle train/valid split only for "train" split
        if self.split == "train" and self.subsplit is not None:
            train_files, valid_files = self._create_train_valid_split(all_wav_files)
            
            if self.subsplit == "train":
                wav_files = train_files
                print(f"Using {len(wav_files)} files for training")
            elif self.subsplit == "valid":
                wav_files = valid_files
                print(f"Using {len(wav_files)} files for validation")
            else:
                raise ValueError(f"Invalid subsplit '{self.subsplit}'. Must be 'train' or 'valid'")
        else:
            wav_files = all_wav_files
            if self.split == "train" and self.subsplit is None:
                print("Note: Using all train files. Use subsplit='train' or 'valid' for train/validation split")
        
        # Process selected files
        for wav_file in wav_files:
            # Get corresponding .PHN file
            phn_file = wav_file.replace('.wav', '.PHN')
            
            if not os.path.exists(phn_file):
                print(f"Warning: No phoneme file found for {wav_file}")
                continue
                
            try:
                # Load audio
                orig_sr, audio = scipy.io.wavfile.read(wav_file)
                audio = torch.from_numpy(audio.astype(np.float64))
                
                # Resample if necessary
                if orig_sr != self.samplerate:
                    resampler = torchaudio.transforms.Resample(orig_sr, self.samplerate)
                    audio = resampler(audio)
                
                audio = audio.squeeze(0)  # Remove channel dimension
                
                # Load phoneme annotations
                phonemes = self._load_phonemes(phn_file, orig_sr)
                
                # Create segments for each phoneme
                for start_sample, end_sample, phoneme in phonemes:
                    # Convert sample indices to new sampling rate
                    start_resample = int(start_sample * self.samplerate / orig_sr)
                    end_resample = int(end_sample * self.samplerate / orig_sr)
                    
                    # Extract audio segment
                    audio_segment = audio[start_resample:end_resample]
                    
                    # Pad or truncate to fixed length
                    if len(audio_segment) > self.sample_length:
                        audio_segment = audio_segment[:self.sample_length]
                    else:
                        audio_segment = F.pad(audio_segment, (0, self.sample_length - len(audio_segment)), "constant", 0)
                    
                    # Map phoneme to index
                    phoneme_idx = self.mapping_phone.get(phoneme, 38)  # 38 for unknown phonemes
                    noise = torch.randn_like(audio_segment) * 1e-6
                    audio_segment = audio_segment + noise
                    self.phone_segments.append({
                        'audio': audio_segment.unsqueeze(0),
                        'label': phoneme_idx,
                        'phoneme': phoneme,
                        'file': os.path.basename(wav_file)
                    })
                
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.phone_segments)} phoneme segments")
        
    def _load_phonemes(self, phn_file: str, sample_rate: int) -> List[Tuple[int, int, str]]:
        """
        Load phoneme annotations from .PHN file
        
        Args:
            phn_file: Path to .PHN file
            sample_rate: Original sample rate
            
        Returns:
            List of (start_sample, end_sample, phoneme) tuples
        """
        phonemes = []
        
        with open(phn_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 3:
                    start_sample = int(parts[0])
                    end_sample = int(parts[1])
                    phoneme = parts[2].lower()
                    
                    phonemes.append((start_sample, end_sample, phoneme))
        
        return phonemes
    
    def get_phoneme_counts(self) -> Dict[str, int]:
        """Get count of each phoneme in the dataset"""
        counts = {}
        for segment in self.phone_segments:
            phoneme = segment['phoneme']
            counts[phoneme] = counts.get(phoneme, 0) + 1
        return counts
    
    def get_unique_phonemes(self) -> List[str]:
        """Get list of unique phonemes in the dataset"""
        return list(set(segment['phoneme'] for segment in self.phone_segments))
    
    def get_split_info(self) -> Dict:
        """Get information about the current split"""
        info = {
            'split': self.split,
            'subsplit': self.subsplit,
            'total_segments': len(self.phone_segments),
            'unique_phonemes': len(self.get_unique_phonemes())
        }
        
        if self.split == "train":
            train_split_file, valid_split_file = self._get_split_file_paths()
            info['train_split_file'] = train_split_file
            info['valid_split_file'] = valid_split_file
            info['split_files_exist'] = os.path.exists(train_split_file) and os.path.exists(valid_split_file)
        
        return info
    
    def __len__(self):
        return len(self.phone_segments)
    
    def __getitem__(self, idx):
        segment = self.phone_segments[idx]
        audio = segment['audio']
        label = segment['label']
        
        return audio, label




# class TimitDataset(Dataset):
#     def __init__(self, data_dir: str, split: str = "train", samplerate: int = 16000, length: float = 0.5,):
#         """
#         Custom TIMIT Dataset Loader
        
#         Args:
#             data_dir: Path to TIMIT root directory
#             split: "train" or "test"
#             samplerate: Target sampling rate
#             length: Length of audio segments in seconds
#         """
#         super().__init__()
        
#         self.data_dir = data_dir
#         self.split = split.lower()
#         self.samplerate = samplerate
#         self.length = length
#         self.sample_length = int(length * samplerate)
        
#         # Phoneme mapping (same as your original)
#         self.mapping_phone = {
#             'iy': 0, 'ih': 1, 'ix': 1, 'eh': 2, 'ae': 3, 'ax': 4, 'ah': 4, 'ax-h': 4,
#             'uw': 5, 'ux': 5, 'uh': 6, 'ao': 7, 'aa': 7, 'ey': 8, 'ay': 9, 'oy': 10,
#             'aw': 11, 'ow': 12, 'er': 13, 'axr': 13, 'l': 14, 'el': 14, 'r': 15,
#             'w': 16, 'y': 17, 'm': 18, 'em': 18, 'n': 19, 'en': 19, 'nx': 19,
#             'ng': 20, 'eng': 20, 'v': 21, 'f': 22, 'dh': 23, 'th': 24, 'z': 25,
#             's': 26, 'zh': 27, 'sh': 27, 'jh': 28, 'ch': 29, 'b': 30, 'p': 31,
#             'd': 32, 'dz': 33, 't': 34, 'g': 35, 'k': 36, 'hh': 37, 'hv': 37,
#         }
        
#         self.phone_list = ['iy', 'ih', 'eh', 'ae', 'ax', 'uw', 'uh', 'ao', 'ey', 'ay', 
#                           'oy', 'aw', 'ow', 'er', 'l', 'r', 'w', 'y', 'm', 'n', 'ng', 
#                           'v', 'f', 'dh', 'th', 'z', 's', 'zh', 'jh', 'ch', 'b', 'p', 
#                           'd', 'dz', 't', 'g', 'k', 'hh', 'sil']
        
#         # Load and process data
#         self.phone_segments = []
#         self._load_data()
        
#     def _load_data(self):
#         """Load all audio files and phoneme annotations"""
#         split_dir = os.path.join(self.data_dir, self.split)
        
#         if not os.path.exists(split_dir):
#             raise ValueError(f"Split directory {split_dir} does not exist")
        
#         print(f"Loading {self.split} data from {split_dir}")
        
#         # Find all .wav files
#         wav_files = glob.glob(os.path.join(split_dir, "**", "*.wav"), recursive=True)
#         print(f"Found {len(wav_files)} audio files")
        
#         for wav_file in wav_files:
#             # Get corresponding .PHN file
#             phn_file = wav_file.replace('.wav', '.PHN')
            
#             if not os.path.exists(phn_file):
#                 print(f"Warning: No phoneme file found for {wav_file}")
#                 continue
                
#             try:
#                 # Load audio
#                 #audio, orig_sr = torchaudio.load(wav_file)
#                 orig_sr, audio = scipy.io.wavfile.read(wav_file)
#                 audio = torch.from_numpy(audio.astype(np.float64))
#                 audio.unsqueeze(0)
#                 # Resample if necessary
#                 if orig_sr != self.samplerate:
#                     resampler = torchaudio.transforms.Resample(orig_sr, self.samplerate)
#                     audio = resampler(audio)
                
#                 #audio = audio.squeeze(0)  # Remove channel dimension
                
#                 # Load phoneme annotations
#                 phonemes = self._load_phonemes(phn_file, orig_sr)
                
#                 # Create segments for each phoneme
#                 for start_sample, end_sample, phoneme in phonemes:
#                     # print(f"{start_sample=}")
#                     # print(f"{end_sample=}")
#                     # print(f"{phoneme=}")
#                     # Convert sample indices to new sampling rate
#                     start_resample = int(start_sample * self.samplerate / orig_sr)
#                     end_resample = int(end_sample * self.samplerate / orig_sr)
                    
#                     # Extract audio segment
#                     audio_segment = audio[start_resample:end_resample]
                    
#                     # Skip very short segments
#                     # if len(audio_segment) < self.samplerate * 0.01:  # Skip < 10ms
#                     #     print("WALLAH C TRO COURT")
#                     #     continue
                    
#                     # Pad or truncate to fixed length
#                     if len(audio_segment) > self.sample_length:
#                         audio_segment = audio_segment[:self.sample_length]
#                     else:
#                         audio_segment = F.pad(audio_segment, (0, self.sample_length - len(audio_segment)), "constant", 0)
                    
#                     # Map phoneme to index
#                     phoneme_idx = self.mapping_phone.get(phoneme, 38)  # 38 for unknown phonemes
                    
#                     # print("Check that shit")
#                     # print(f"{audio.shape=}")
#                     # print(f"{phoneme_idx=}")
#                     # print(f"{phoneme=}")

#                     self.phone_segments.append({
#                         'audio': audio_segment,
#                         'label': phoneme_idx,
#                         'phoneme': phoneme,
#                         'file': os.path.basename(wav_file)
#                     })
                
#             except Exception as e:
#                 print(f"Error processing {wav_file}: {e}")
#                 continue
        
#         print(f"Successfully loaded {len(self.phone_segments)} phoneme segments")
        
#     def _load_phonemes(self, phn_file: str, sample_rate: int) -> List[Tuple[int, int, str]]:
#         """
#         Load phoneme annotations from .PHN file
        
#         Args:
#             phn_file: Path to .PHN file
#             sample_rate: Original sample rate
            
#         Returns:
#             List of (start_sample, end_sample, phoneme) tuples
#         """
#         phonemes = []
        
#         with open(phn_file, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
                    
#                 parts = line.split()
#                 if len(parts) >= 3:
#                     start_sample = int(parts[0])
#                     end_sample = int(parts[1])
#                     phoneme = parts[2].lower()
                    
#                     phonemes.append((start_sample, end_sample, phoneme))
        
#         return phonemes
    
#     def get_phoneme_counts(self) -> Dict[str, int]:
#         """Get count of each phoneme in the dataset"""
#         counts = {}
#         for segment in self.phone_segments:
#             phoneme = segment['phoneme']
#             counts[phoneme] = counts.get(phoneme, 0) + 1
#         return counts
    
#     def get_unique_phonemes(self) -> List[str]:
#         """Get list of unique phonemes in the dataset"""
#         return list(set(segment['phoneme'] for segment in self.phone_segments))
    
#     def __len__(self):
#         return len(self.phone_segments)
    
#     def __getitem__(self, idx):
#         segment = self.phone_segments[idx]
#         audio = segment['audio']
#         label = segment['label']
        
#         return audio, label














# class TimitDataset(Dataset):

#     def __init__(self,mode="TRAIN",samplerate=16000,length=0.5,dataset_utterance=None):
#         super().__init__()
#         dataset = load_dataset('timit_asr', data_dir='/path/to/TIMIT',split=mode)
#         #dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split=mode)
#         self.mapping_phone = {
#                 'iy':0,
#                 'ih':1,
#                 'ix':1,
#                 'eh':2,
#                 'ae':3,
#                 'ax':4,
#                 'ah':4,
#                 'ax-h':4,
#                 'uw':5,
#                 'ux':5,
#                 'uh':6,
#                 'ao':7,
#                 'aa':7,
#                 'ey':8,
#                 'ay':9,
#                 'oy':10,
#                 'aw':11,
#                 'ow':12,
#                 'er':13,
#                 'axr':13,
#                 'l':14,
#                 'el':14,
#                 'r':15,
#                 'w':16,
#                 'y':17,
#                 'm':18,
#                 'em':18,
#                 'n':19,
#                 'en':19,
#                 'nx':19,
#                 'ng':20,
#                 'eng':20,
#                 'v':21,
#                 'f':22,
#                 'dh':23,
#                 'th':24,
#                 'z':25,
#                 's':26,
#                 'zh':27,
#                 'sh':27,
#                 'jh':28,
#                 'ch':29,
#                 'b':30,
#                 'p':31,
#                 'd':32,
#                 'dz':33,
#                 't':34,
#                 'g':35,
#                 'k':36,
#                 'hh':37,
#                 'hv':37,
#         }
        
#         dataset = dataset.cast_column("audio", Audio(sampling_rate=samplerate))
        
#         dataset = dataset.remove_columns(["text","word_detail","dialect_region","sentence_type","speaker_id","id"])
#         print("Cleaning...",end="")
#         print("..",end="")
        
#         phonemes = []
#         for detail in dataset['phonetic_detail']:
#             for p in detail['utterance']:
#                 if p not in phonemes:
#                     phonemes.append(p)

#         self.list_phone = ['iy','ih','eh','ae','ax','uw','uh','ao','ey','ay','oy','aw','ow','er','l','r','w','y','m','n','ng','v','f','dh','th','z','s','zh','jh','ch','b','p','d','dz','t','g','k','hh','sil']
        
        
#         dataset = dataset.map(self.to_index,input_columns='phonetic_detail')
        
#         dataset.set_format(type='torch', columns=['audio', 'phonetic_detail','start','stop'])
        
#         print("..OK")
#         self.samplerate = samplerate
#         self.length=length
#         self.sample_length=int(length*samplerate)
#         print("Dataset preparation")
#         self.phone = []
#         for e in dataset:
            
#             for label,start,stop in zip(e['phonetic_detail'],e['start'],e['stop']):
#                 audio = e['audio']['array'][start:stop]
#                 self.phone.append(
#                     {'audio':F.pad(audio[:self.sample_length],(0,max(0,self.sample_length-audio.shape[0])), "constant", 0),
#                      'label':label
#                     }
#                 )
        
        
        
#     def to_index(self,example):
#         dicti = {"phonetic_detail":[]}
        
#         dicti = {'start':example['start'],'stop':example['stop'],'phonetic_detail': [self.mapping_phone[x] if x in self.mapping_phone else 38 for x in example['utterance']]}
#         return dicti
    
#     def __getitem__(self,idx):
#         sample = self.phone.__getitem__(idx)
#         label = sample['label']
#         audio = sample['audio']
        
#         #audio = F.pad(audio[:self.sample_length],(0,max(0,self.sample_length-audio.shape[0])), "constant", 0)
#         return audio,label
#     def __len__(self):
#         return len(self.phone)