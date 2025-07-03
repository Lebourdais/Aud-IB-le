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
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ssl_downstream_timit'),
    'sample_rate': 16000,
    'optim': {
        'epochs': 80,
        'batch_size': 64,  
        'learning_rate': 0.001,
        'patience_scheduler': 200,
    },
    'data': {
      "dataset_name": "timit",
      "train": {
        "data_dir": "/lium/corpus/base/TIMIT/",
        "split": "train",
        "subsplit": "train",
        "length": 0.5,},
      "valid": {
            "data_dir": "/lium/corpus/base/TIMIT/",
            "split": "train",
            "subsplit": "valid",
            "length": 0.5,},
    },
    'model': {
      "encoder_type": "wavlm",
      "freeze": True,
      "num_classes": 39,
      "pooling_method": "mean"
    },
    
    "job_id": job_id,
    "cluster": cluster
}
conf = {
  "001": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [0]
      },
    },
  "002": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [1]
      },
    },
  "003": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [2]
      },
    },
  "004": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [3]
      },
    },
  "005": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [4]
      },
    },
  "006": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [5]
      },
    },
  "007": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [6]
      },
    },
  "008": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [7]
      },
    },
  "009": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [8]
      },
    },
  "010": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [9]
      },
    },
  "011": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [10]
      },
    },
  "012": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [11]
      },
    },
  "013": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [12]
      },
    },
  "014": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [13]
      },
    },
  "015": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [0]
      },
    },
  "016": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 50,
        "pooling_method": "mean",
        "use_layer_indices": [1]
      },
    },
  "017": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [2]
      },
    },
  "018": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [3]
      },
    },
  "019": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [4]
      },
    },
  "020": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [5]
      },
    },
  "021": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [6]
      },
    },
  "022": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [7]
      },
    },
  "023": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [8]
      },
    },
  "024": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [9]
      },
    },
  "025": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [10]
      },
    },
  "026": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [11]
      },
    },
  "027": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [12]
      },
    },
  "028": {
      'model': {
        "encoder_type": "wavlm",
        "freeze": True,
        "num_classes": 39,
        "pooling_method": "mean",
        "use_layer_indices": [13]
      },
    },
}