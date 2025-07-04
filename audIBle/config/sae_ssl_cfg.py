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
    #'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ssl'),
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ssl'),
    'sample_rate': 16000,
    'optim': {
        'epochs': 100,
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
      "sae_dim": 2048,
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
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [2,7,12]
        },
    },
  "002": {
        'model': {
        "encoder_type": "beats",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [2,7,12]
        },
    },
  "003": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [2,7,12]
        },
    },
  "004": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [2,7,12]
        },
    },
  # AST appropriate layers
  "005": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.95,
        "freeze": True,
        "layer_indices": [10, 12]
        },
    },
  "006": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [10, 12]
        },
    },
  "007": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.85,
        "freeze": True,
        "layer_indices": [10, 12]
        },
    },
  "008": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.8,
        "freeze": True,
        "layer_indices": [10, 12]
        },
    },
  "009": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.75,
        "freeze": True,
        "layer_indices": [10, 12]
        },
    },
  # AST appropriate layers wiyth pooling
  "010": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.95,
        "freeze": True,
        "layer_indices": [10, 12],
        "pooling_method": "mean"
        },
    },
  "011": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [10, 12],
        "pooling_method": "mean"
        },
    },
  "012": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.85,
        "freeze": True,
        "layer_indices": [10, 12],
        "pooling_method": "mean"
        },
    },
  "013": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.8,
        "freeze": True,
        "layer_indices": [10, 12],
        "pooling_method": "mean"
        },
    },
  "014": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.75,
        "freeze": True,
        "layer_indices": [10, 12],
        "pooling_method": "mean"
        },
    },
  }
  # # HuBERT appropriate layers
  # "010": {
  #       'model': {
  #       "encoder_type": "hubert",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.95,
  #       "freeze": True,
  #       "layer_indices": [2, 5]
  #       },
  #   },
  # "011": {
  #       'model': {
  #       "encoder_type": "hubert",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.9,
  #       "freeze": True,
  #       "layer_indices": [2, 5]
  #       },
  #   },

  # "012": {
  #       'model': {
  #       "encoder_type": "hubert",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.85,
  #       "freeze": True,
  #       "layer_indices": [2, 5]
  #       },
  #    },
  # "013": {
  #       'model': {
  #       "encoder_type": "hubert",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.8,
  #       "freeze": True,
  #       "layer_indices": [2, 5]
  #       },
  #   },
  # "014": {
  #       'model': {
  #       "encoder_type": "hubert",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.75,
  #       "freeze": True,
  #       "layer_indices": [2, 5]
  #       },
  #   },
  # # WavLM appropriate layers
  # "015": {
  #       'model': {
  #       "encoder_type": "wavlm",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.95,
  #       "freeze": True,
  #       "layer_indices": [1, 2]
  #       },
  #   },
  # "016": {
  #       'model': {
  #       "encoder_type": "wavlm",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.9,
  #       "freeze": True,
  #       "layer_indices": [1, 2]
  #       },
  #   },
  # "017": {
  #       'model': {
  #       "encoder_type": "wavlm",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.85,
  #       "freeze": True,
  #       "layer_indices": [1, 2]
  #       },
  #   },
  # "018": {
  #       'model': {
  #       "encoder_type": "wavlm",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.8,
  #       "freeze": True,
  #       "layer_indices": [1, 2]
  #       },
  #   },
  # "019": {
  #       'model': {
  #       "encoder_type": "wavlm",
  #       "sae_method": "top-k",
  #       "sae_dim": 2048,
  #       "sparsity": 0.75,
  #       "freeze": True,
  #       "layer_indices": [1, 2]
  #       },
  #   },
