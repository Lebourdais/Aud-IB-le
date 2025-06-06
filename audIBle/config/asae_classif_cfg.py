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
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/sparse_classif'),
    'sample_rate': 22050,
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.001,
        'patience_early_stop': 15,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
    },
    'data': {
      'root': '/lium/corpus/vrac/audio_tagging/urbansound8k/urbansound8k',
      'folds_train': [1,2,3,4,5,6,7,8],
      'folds_valid': [9],
    },
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/006_spec_autoencoder_urbasound8k_42"), 
    },
    
    "job_id": job_id,
    "cluster": cluster
}

conf = {
  "001": {
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/006_spec_autoencoder_urbasound8k_42"), # 80% sparsity
    },
  },
  "002":{
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/007_spec_autoencoder_urbasound8k_42"), # 75% sparsity
    },
  },
  "003":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
        }, 
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/009_spec_autoencoder_urbasound8k_42"), # 90% sparsity
    },
  },
  "004":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.0005,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
    },
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/007_spec_autoencoder_urbasound8k_42"), 
    },
  },
  "005":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
    },
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/007_spec_autoencoder_urbasound8k_42"), # 75% sparsity
    },
  },
  "006":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
        }, 
    'model': {     
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/010_spec_autoencoder_urbasound8k_42"), # 95% sparsity
    },
  },
  "007":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
        }, 
    'model': {     
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/008_spec_autoencoder_urbasound8k_42"), # 85% sparsity
    },
  },
  "008":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.1) # BCE, MSE Spectro
        }, 
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": False
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/009_spec_autoencoder_urbasound8k_42"), 
    },
  },
  "009":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.5) # BCE, MSE Spectro
        }, 
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": False
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/009_spec_autoencoder_urbasound8k_42"), 
    },
  },
  "010":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 2.0) # CE, MSE Spectro
        }, 
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": False
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/009_spec_autoencoder_urbasound8k_42"), 
    },
  },
  "011":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # CE, MSE Spectro
        }, 
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True,
            "use_seq_modeling": True,
            "tcn_bn_dim": 512,
            "tcn_hid_dim": 256,
            "tcn_n_repeat": 2,
            "tcn_n_block": 2,
            "pool_type": None,
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/009_spec_autoencoder_urbasound8k_42"), 
    },
  },
  "012":{
    'optim': {
        'epochs': 200,
        'batch_size': 64,  
        'learning_rate': 0.001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # CE, MSE Spectro
        }, 
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True,
            "use_seq_modeling": False,
            "pool_type": "avg",
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/009_spec_autoencoder_urbasound8k_42"), #90% sparsity 
    },
  },
  "013":{
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True,
            "use_seq_modeling": False,
            "pool_type": "avg"
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/007_spec_autoencoder_urbasound8k_42"), # 75% sparsity
    },
  },
  "014":{
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True,
            "use_seq_modeling": False,
            "pool_type": "att"
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/011_spec_autoencoder_urbasound8k_42"), # 50% sparsity
    },
  },
  "015":{
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True,
            "use_seq_modeling": False,
            "pool_type": "att"
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/012_spec_autoencoder_urbasound8k_42"), # 0% sparsity
    },
  },
  "016":{
    'optim': {
        'epochs': 50,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
    },
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/007_spec_autoencoder_urbasound8k_84"), # 75% sparsity
    },
  },
  "017": {
    'optim': {
        'epochs': 50,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
        }, 
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/006_spec_autoencoder_urbasound8k_84"), # 80% sparsity
    },
  },
  "018":{
    'optim': {
        'epochs': 50,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
        }, 
    'model': {     
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/008_spec_autoencoder_urbasound8k_84"), # 85% sparsity
    },
  },
  "019":{
    'optim': {
        'epochs': 50,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # CE, MSE Spectro
        }, 
    'model': {
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True,
            "use_seq_modeling": False,
            "pool_type": "avg",
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/009_spec_autoencoder_urbasound8k_84"), #90% sparsity 
    },
  },
  "020":{
    'optim': {
        'epochs': 50,
        'batch_size': 64,  
        'learning_rate': 0.0001,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
        }, 
    'model': {     
        "classifier": {
            "n_classes": 10,
            "hidden_dim": 1024,
            "method": "top-k",
            "freeze_autoencoder": True
        },
        "asae_exp_path": os.path.join(EXP_ROOT,"train/SAE/sparse_audio_ae/010_spec_autoencoder_urbasound8k_84"), # 95% sparsity
    },
  },

}
