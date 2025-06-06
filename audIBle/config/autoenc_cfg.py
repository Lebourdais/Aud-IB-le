import os
import random
import socket
import string

cluster = socket.gethostname()
slurm = "SLURM_JOB_ID" in os.environ

EXP_ROOT = os.environ["EXP_ROOT"]

if slurm:
  job_id = os.environ["SLURM_JOB_ID"]
else:
  job_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))

common_parameters = {
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/sparse_audio_ae'),
    'sample_rate': 22050,
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.001,
        'patience_early_stop': 15,
        'patience_scheduler': 200,
    },
    'data': {
      'root': '/lium/corpus/vrac/audio_tagging/urbansound8k/urbansound8k',
      'folds_train': [1,2,3,4,5,6,7,8],
      'folds_valid': [9],
    },
    'model': {
        'normalize_audio': True,
        "scale": "lin",
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "center": True, 
        "pad": 0,
        "latent_dim": 256,
        "input_channels": 1,
        "hid_dims": [16, 32, 64, 128],
        "use_attention": False,
        "activation_slope": 0.2,
        "attention_dim": 64,
        "sparsity": None,
        "sparse_method": None
    },
    "job_id": job_id,
    "cluster": cluster
}

conf = {
    "001": {
        "model": {
          "scale": "log_biais"
        },
    },
    "002": {
        "model": {
            "scale": "db"
        },
    },
    "003": {
        "model": {
            "scale": "lin",
        },
    },
    "004": {
        "model": {
            "scale": "log",
        },
    },
    "005": {
        "model": {
            "scale": "log",
        },
    },
    "006": {
        "model": {
            "scale": "log",
            "sparsity": 0.8,
            "sparse_method": "top-k",
            "latent_dim": 1024,
        },
    },
    "007": {
        "model": {
            "scale": "log",
            "sparsity": 0.75,
            "sparse_method": "top-k",
            "latent_dim": 1024,
        },
    },
    "008": {
        "model": {
            "scale": "log",
            "sparsity": 0.85,
            "sparse_method": "top-k",
            "latent_dim": 1024,
        },
    },
    "009": {
        "model": {
            "scale": "log",
            "sparsity": 0.9,
            "sparse_method": "top-k",
            "latent_dim": 1024,
        },
    },
    "010": {
        "model": {
            "scale": "log",
            "sparsity": 0.95,
            "sparse_method": "top-k",
            "latent_dim": 1024,
        },
    },
    "011": {
        "model": {
            "scale": "log",
            "sparsity": 0.5,
            "sparse_method": "top-k",
            "latent_dim": 1024,
        },
    },
    "012": {
        "model": {
            "scale": "log",
            "sparsity": 0.0,
            "sparse_method": "top-k",
            "latent_dim": 1024,
        },
    },
}
