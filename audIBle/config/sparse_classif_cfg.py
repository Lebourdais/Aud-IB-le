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
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ae_debug'),
    'sample_rate': 22050,
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.001,
        'patience_early_stop': 15,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.5, 0.0) # BCE, MSE SAE, MSE Spectro
    },
    'data': {
      'root': '/lium/corpus/vrac/tmario/sed/urbansound8k/urbansound8k',
      'folds_train': [1,2,3,4,5,6,7,8],
      'folds_valid': [9],
    },
    'model': {
        "autoencoder": 
            {
            'normalize_audio': True,
            "scale": "log",
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
            "attention_dim": 64
        },
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 256,
            "sae_dim": 1024,
            "sparsity": 0.05,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "ae_ckpt_path": os.path.join(EXP_ROOT,"train/SAE/ae_debug/004_spec_autoencoder_urbasound8k_42", "best_model.pth")
    },
    
    "job_id": job_id,
    "cluster": cluster
}

conf = {
  "001": {
    'model': {
        "autoencoder": 
            {
            'normalize_audio': True,
            "scale": "log",
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
            "attention_dim": 64
        },
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 256,
            "sae_dim": 1024,
            "sparsity": 0.05,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "ae_ckpt_path": os.path.join(EXP_ROOT,"train/SAE/ae_debug/004_spec_autoencoder_urbasound8k_42", "best_model.pth")
    },
  },
  "002":{
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 256,
            "sae_dim": 1024,
            "sparsity": 0.1,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "ae_ckpt_path": os.path.join(EXP_ROOT,"train/SAE/ae_debug/004_spec_autoencoder_urbasound8k_42", "best_model.pth")
    },
  },
  "003":{
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 256,
            "sae_dim": 1024,
            "sparsity": 0.15,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "ae_ckpt_path": os.path.join(EXP_ROOT,"train/SAE/ae_debug/004_spec_autoencoder_urbasound8k_42", "best_model.pth")
    },
  }
  
}