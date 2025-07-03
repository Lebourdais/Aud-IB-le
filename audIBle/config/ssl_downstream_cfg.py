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
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ssl_downstream'),
    'sample_rate': 16000,
    'optim': {
        'epochs': 80,
        'batch_size': 16,  
        'learning_rate': 0.001,
        'patience_scheduler': 200,
    },
    'data': {
      "dataset_name": "esc50",
      "train": {
        "root": "/lium/corpus/vrac/audio_tagging/",
        "part": "train",
        "target_samplerate": 16000},
      "valid": {
        "root": "/lium/corpus/vrac/audio_tagging/",
        "part": "valid",
        "target_samplerate": 16000
      },    
    },
    'model': {
      "encoder_type": "wavlm",
      "freeze": True,
      "num_classes": 50,
      "pooling_method": "mean"
    },
    
    "job_id": job_id,
    "cluster": cluster
}
conf = {
  "001": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean"
      },
    },
  "002": {
      'model': {
        "encoder_type": "beats",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean"
      },
    },
  "003": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean"
      },
    },
  "004": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean"
      },
    },
  "005": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [0]
      },
    },
  "006": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [1]
      },
    },
  "007": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [2]
      },
    },
  "008": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [3]
      },
    },
  "009": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [4]
      },
    },
  "010": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [5]
      },
    },
  "011": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [6]
      },
    },
  "012": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [7]
      },
    },
  "013": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [8]
      },
    },
  "014": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [9]
      },
    },
  "015": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [10]
      },
    },
  "016": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [11]
      },
    },
  "017": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [12]
      },
    },
  "018": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [13]
      },
    },
  "019": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [0]
      },
    },
  "020": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [1]
      },
    },
  "021": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [2]
      },
    },
  "022": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [3]
      },
    },
  "023": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [4]
      },
    },
  "024": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [5]
      },
    },
  "025": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [6]
      },
    },
  "026": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [7]
      },
    },
  "027": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [8]
      },
    },
  "028": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [9]
      },
    },
  "029": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [10]
      },
    },
  "030": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [11]
      },
    },
  "031": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [12]
      },
    },
  "032": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [13]
      },
    },
  "033": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [0]
      },
    },
  "034": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [1]
      },
    },
  "035": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [2]
      },
    },
  "036": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [3]
      },
    },
  "037": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [4]
      },
    },
  "038": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [5]
      },
    },
  "039": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [6]
      },
    },
  "040": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [7]
      },
    },
  "041": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [8]
      },
    },
  "042": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [9]
      },
    },
  "043": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [10]
      },
    },
  "044": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [11]
      },
    },
  "045": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [12]
      },
    },
  "046": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [13]
      },
    },
}