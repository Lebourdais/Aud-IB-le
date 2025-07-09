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
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ssl_downstream_vocalset'),
    'sample_rate': 16000,
    'optim': {
        'epochs': 1,
        'batch_size': 32,  
        'learning_rate': 0.001,
        'patience_scheduler': 200,
    },
    'data': {
      "dataset_name": "vocalset",
      "train": {
        "root": "/lium/corpus/vrac/VocalSet/FULL",
        "split": "train",
        "seed": 42,
        "ratio": 0.8,
        "duration": 5.0,
        "target_sr": 16000,},
      "valid": {
        "root": "/lium/corpus/vrac/VocalSet/FULL",
        "split": "valid",
        "seed": 42,
        "ratio": 0.8, # ratio is the amount of train data. The <split> parameter allows to select the train files or the valid files
        "duration": 5.0,
        "target_sr": 16000,},
    },
    'model': {
      "encoder_type": "wavlm",
      "freeze": True,
      "num_classes": 10,
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
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [0]
      },
    },
  "002": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [1]
      },
    },
  "003": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [2]
      },
    },
  "004": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [3]
      },
    },
  "005": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [4]
      },
    },
  "006": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [5]
      },
    },
  "007": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [6]
      },
    },
  "008": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [7]
      },
    },
  "009": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [8]
      },
    },
  "010": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [9]
      },
    },
  "011": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [10]
      },
    },
  "012": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [11]
      },
    },
  "013": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [12]
      },
    },
  "014": {
      'model': {
        "encoder_type": "hubert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [13]
      },
    },
  "015": {
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [0]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "016": {
      'sample_rate': 24000,
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [1]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "017": {
      'sample_rate': 24000,
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [2]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "018": {
      'sample_rate': 24000,
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [3]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "019": {
      'sample_rate': 24000,    
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [4]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "020": {
      'sample_rate': 24000,
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [5]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "021": {
      'sample_rate': 24000,    
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [6]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "022": {
      'sample_rate': 24000,
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [7]
      },
    },
  "023": {
      'sample_rate': 24000,
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [8]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "024": {
      'sample_rate': 24000,    
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [9]
      },
    },
  "025": {
      'sample_rate': 24000,    
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [10]
      },
    },
  "026": {
      'sample_rate': 24000,
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [11]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "027": {
      'sample_rate': 24000,
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [12]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "028": {
      'sample_rate': 24000,    
      'model': {
        "encoder_type": "mert",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [13]
      },
      'data': {
        "train": {
          "target_sr": 24000,},
        "valid": {
          "target_sr": 24000,},
      },
    },
  "029": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [0]
      },
    },
  "030": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [1]
      },
    },
  "031": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [2]
      },
    },
  "032": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [3]
      },
    },
  "033": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [4]
      },
    },
  "034": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [5]
      },
    },
  "035": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [6]
      },
    },
  "036": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [7]
      },
    },
  "037": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [8]
      },
    },
  "038": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [9]
      },
    },
  "039": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [10]
      },
    },
  "040": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [11]
      },
    },
  "041": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [12]
      },
    },
  "042": {
      'model': {
        "encoder_type": "ast",
        "freeze": True,
        "num_classes": 10,
        "pooling_method": "mean",
        "use_layer_indices": [13]
      },
    },
}