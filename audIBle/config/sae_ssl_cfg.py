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
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ssl'),
    'sample_rate': 16000,
    'optim': {
        'epochs': 200,
        'batch_size': 16,  
        'learning_rate': 0.001,
        'patience_early_stop': 15,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
    },
    'data': {
      "dataset_name": "esc50",
      "train": {
        "root": "/lium/corpus/vrac/audio_tagging/",
        "part": "train"},
      "valid": {
        "root": "/lium/corpus/vrac/audio_tagging/",
        "part": "valid"
      },    
    },
    'model': {
      "encoder_type": "wavlm",
      "sae_method": "top-k",
      "sae_dim": 1024,
      "sparsity": 0.9,
      "freeze": True,
      "layer_indices": [5,6,7,8]
    },
    
    "job_id": job_id,
    "cluster": cluster
}
conf = {
  "001": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 1024,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [5]
        },
    },
  "002": {
        'model': {
        "encoder_type": "beats",
        "sae_method": "top-k",
        "sae_dim": 1024,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [5]
        },
    },
  "003": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 1024,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [5]
        },
    },
  "004": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 1024,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [5]
        },
    },
}