import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_url
from tqdm import tqdm
import pandas as pd
import torchaudio
import os
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import IterableDataset
import shutil



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

class WHAMDataset(IterableDataset):
    """Implements class for WHAM! dataset.

    Arguments
    ---------
    data_dir: str or Path
        Directory where the dataset is stored.
    target_length: int
        Expected audio sample length. Used for padding and cropping.
    sample_rate: int
        Sample rate of the audio samples.
    """

    def __init__(self, data_dir, target_length=4, sample_rate=22050):
        self.data_dir = data_dir
        self.target_length = target_length
        self.sample_rate = sample_rate

        # Get a list of all WAV files in the WHAM data directory
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

    def generate(self):
        """Generates viable audio sample from the WHAM set."""
        while True:
            idx = np.random.choice([i for i in range(len(self.file_list))])
            file_path = os.path.join(self.data_dir, self.file_list[idx])

            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.mean(0, keepdim=True)

            # Resample if needed
            if self.sample_rate != sr:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(
                    waveform
                )

            # Cut audio to the target length
            if waveform.shape[1] > self.target_length * self.sample_rate:
                start = 0
                end = int(self.target_length * self.sample_rate)
                waveform = waveform[:, start:end]

            zeros = (
                int(self.target_length * self.sample_rate) - waveform.shape[1]
            )
            waveform = F.pad(waveform, (0, zeros))

            yield waveform

    def __iter__(self):
        """Iterator constructor."""
        return iter(self.generate())


def combine_batches(clean, noise_loader):
    """Combines waveforms at 0dB.

    Arguments
    ---------
    clean: torch.Tensor
        Original sample.
    noise_loader: int
        DataLoader for the contamination dataset.

    Returns
    -------
    Mixture : torch.Tensor
    """
    batch_size = clean.shape[0]

    noise = []
    for _ in range(batch_size):
        noise.append(next(noise_loader))
    noise = torch.stack(noise).to(clean.device)

    if noise.ndim == 3:
        noise = noise.squeeze(1)
    elif noise.ndim == 1:
        noise = noise[None]

    clean_l2 = (clean**2).sum(-1) ** 0.5
    noise_l2 = (noise**2).sum(-1) ** 0.5

    # Combine the batches at 0dB
    combined_batch = clean / clean_l2[..., None] + noise / noise_l2[..., None]
    combined_batch = (
        combined_batch / torch.max(combined_batch, dim=1, keepdim=True).values
    )

    return combined_batch


def download_wham(wham_path: str):
    """
    This function automatically downloads the WHAM! dataset to the specified data path in the wham_path variable

    Arguments
    ---------
    wham_path: str or Path
        Directory used to save the dataset.

    Returns
    -------
    None
    """
    if len(os.listdir(wham_path)) != 0:
        return

    print("WHAM! is missing. Downloading WHAM!. This will take a while...")
    os.makedirs(wham_path, exist_ok=True)

    temp_path = os.path.join(wham_path, "temp_download_wham")

    # download the data
    # fetch(
    #     "wham_noise.zip",
    #     "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com",
    #     savedir=temp_path,
    # )

    # unpack the .zip file
    shutil.unpack_archive(os.path.join(temp_path, "wham_noise.zip"), wham_path)

    files = os.listdir(os.path.join(wham_path, "WHAM", "wham_noise"))
    for fl in files:
        shutil.move(
            os.path.join(wham_path, "WHAM", "wham_noise", fl), wham_path
        )

    # remove the unused datapath
    shutil.rmtree(temp_path)
    shutil.rmtree(os.path.join(wham_path, "WHAM"))

    print(f"WHAM! is downloaded in {wham_path}")


def prepare_wham(
    wham_folder, add_wham_noise, sample_rate, signal_length_s, wham_audio_folder
):
    """Creates WHAM! dataset when needed.

    Arguments
    ---------
    wham_folder: str or Path
        Directory where the dataset is stored.
        If empty, data will be automatically downloaded.
    add_wham_noise: bool
        True when wham contamination is required. When False, returns None.
    sample_rate: int
        Sample rate for the mixture.
    signal_length_s: int
        Seconds. Expected length of the audio sample.
    wham_audio_folder: str or Path
        Points to the wham split. E.g. wham_noise/tr

    Returns
    -------
    WHAM Loader or None, depending on configuration. : WHAMDataset
    """
    if wham_folder is None:
        if add_wham_noise:
            raise Exception("You should specify wham_folder to add noise.")
        return None

    if add_wham_noise:
        # download WHAM! in specified folder
        download_wham(wham_folder)

        dataset = WHAMDataset(
            data_dir=wham_audio_folder,
            target_length=signal_length_s,
            sample_rate=sample_rate,
        )

        return dataset

    return None

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

    def __init__(self, root, part: str = "train", reading_transformations: nn.Module = None):
        super().__init__(root)
        self.part = part
        self._load_meta()

        self.data = []
        self.targets = []
        self.pre_transformations = reading_transformations
        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            wav, sr = torchaudio.load(file_path)
            wav = wav if not self.pre_transformations else torch.Tensor(self.pre_transformations(wav).data)

            self.data.append(wav)
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
            tuple: (image, target) where target is index of the target class.
        """
        mel_spec, target = self.data[index], self.targets[index]
        return mel_spec, target

    def __len__(self):
        return len(self.data)

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
