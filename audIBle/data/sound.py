import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_url
from tqdm import tqdm
import pandas as pd
import torchaudio
import os
import torch.nn as nn

class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = True):
        self.root = os.path.expanduser(root)
        if download:
            self.download()

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


# class ESC_50(AudioDataset):
#     base_folder = 'ESC-50-master'
#     url = "https://codeload.github.com/karolpiczak/ESC-50/zip/master"
#     filename = "ESC-50-master.zip"
#     zip_md5 = '70cce0ef1196d802ae62ce40db11b620'
#     num_files_in_dir = 2000
#     audio_dir = 'audio'
#     label_col = 'category'
#     file_col = 'filename'
#     meta = {
#         'filename': 'meta/esc50.csv',
#         'md5': '54a0d0055a10bb7df84ad340a148722e',
#     }

#     def __init__(self, root, part: str = "train", target_samplerate = 16000, reading_transformations: nn.Module = None):
#         super().__init__(root)
#         self.part = part
#         self._load_meta()
#         self.data = []
#         self.targets = []
#         self.pre_transformations = reading_transformations
#         for _, row in tqdm(self.df.iterrows()):
#             file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
#             wav, sr = torchaudio.load(file_path)
#             wav = wav if not self.pre_transformations else torch.Tensor(self.pre_transformations(wav).data)

#             self.data.append(wav)
#             self.targets.append(self.class_to_idx[row[self.label_col]])
#         if sr != target_samplerate:
#             self.resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_samplerate)

#     def _load_meta(self):
#         path = os.path.join(self.root, self.base_folder, self.meta['filename'])
#         if not check_integrity(path, self.meta['md5']):
#             raise RuntimeError('Dataset metadata file not found or corrupted.' +
#                                ' You can use download=True to download it')

#         data = pd.read_csv(path)
#         if self.part == 'train':
#             folds = [1,2,3]
#         elif self.part == 'valid':
#             folds = [4]
#         else:
#             folds = [5]
#         index = data['fold'].isin(folds)
#         self.df = data[index]
#         self.class_to_idx = {}
#         self.classes = sorted(self.df[self.label_col].unique())
#         for i, category in enumerate(self.classes):
#             self.class_to_idx[category] = i

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         mel_spec, target = self.data[index], self.targets[index]
#         return mel_spec, target

#     def __len__(self):
#         return len(self.data)

#     def _check_integrity(self):
#         path = os.path.join(self.root, self.base_folder, self.meta['filename'])
#         if not check_integrity(path, self.meta['md5']):
#             return False
#         path = os.path.join(self.root, self.base_folder, self.audio_dir)
#         if len(next(os.walk(path))[2]) != self.num_files_in_dir:
#             return False
#         return True

#     def download(self):
#         if self._check_integrity():
#             print('Files already downloaded and verified')
#             return

#         download_url(self.url, self.root, self.filename, self.zip_md5)
        
#         # extract file
#         from zipfile import ZipFile
#         with ZipFile(os.path.join(self.root, self.filename), 'r') as zip:
#             zip.extractall(path=self.root)

class ESC_50(AudioDataset):
    base_folder = 'ESC-50-master'
    url = "https://codeload.github.com/karolpiczak/ESC-50/zip/master"
    filename = "ESC-50-master.zip"
    zip_md5 = '70cce0ef1196d802ae62ce40db11b620'
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': 'meta/esc50.csv',
        'md5': '54a0d0055a10bb7df84ad340a148722e',
    }

    def __init__(self, root, part: str = "train", target_samplerate = 16000, reading_transformations: nn.Module = None, return_path: bool = False):
        super().__init__(root)
        self.part = part
        self.target_samplerate = target_samplerate
        self.reading_transformations = reading_transformations
        self.return_path = return_path
        self._load_meta()
        
        # Store file paths instead of loading all audio into memory
        self.file_paths = []
        self.targets = []
        
        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            self.file_paths.append(file_path)
            self.targets.append(self.class_to_idx[row[self.label_col]])

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')

        data = pd.read_csv(path)
        if self.part == 'train':
            folds = [1,2,3]
        elif self.part == 'valid':
            folds = [4]
        else:
            folds = [5]
        index = data['fold'].isin(folds)
        self.df = data[index]
        self.class_to_idx = {}
        self.classes = sorted(self.df[self.label_col].unique())
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (waveform, target) where waveform is the raw audio and target is index of the target class.
        """
        file_path = self.file_paths[index]
        target = self.targets[index]
        
        # Load raw waveform
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if necessary
        if sample_rate != self.target_samplerate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_samplerate)
            waveform = resampler(waveform)
        
        # Apply any pre-transformations
        if self.reading_transformations:
            waveform = self.reading_transformations(waveform)
        
        # Convert to mono if stereo (ESC-50 should already be mono, but just in case)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        

        if self.return_path:
            return waveform, target, file_path    
        return waveform, target

    def __len__(self):
        return len(self.file_paths)

    def _check_integrity(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            return False
        path = os.path.join(self.root, self.base_folder, self.audio_dir)
        if len(next(os.walk(path))[2]) != self.num_files_in_dir:
            return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.zip_md5)
        
        # extract file
        from zipfile import ZipFile
        with ZipFile(os.path.join(self.root, self.filename), 'r') as zip:
            zip.extractall(path=self.root)


class UrbanSound8k(Dataset):
    def __init__(self, csv_path, audio_dir, folds_to_use=[1], sample_rate=22050, duration=4, transform=None):
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.folds_to_use = folds_to_use
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform

        self.metadata = pd.read_csv(self.csv_path)
        self.metadata = self.metadata[self.metadata['fold'].isin(folds_to_use)]

        self.fixed_length = int(self.sample_rate * self.duration) if duration is not None else None
        self.classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
                        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.audio_dir, f"fold{row['fold']}", row['slice_file_name'])
        label = row['classID']

        waveform, sr = torchaudio.load(file_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or truncate
        if self.fixed_length is not None:
            if waveform.shape[1] < self.fixed_length:
                pad_size = self.fixed_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))
            else:
                waveform = waveform[:, :self.fixed_length]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label