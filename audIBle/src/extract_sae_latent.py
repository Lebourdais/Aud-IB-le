import torch
import numpy as np
import os
import tqdm
import time

from audIBle.data.datasets import select_dataset
from audIBle.nn.utils import count_parameters
from audIBle.nn.sae import SaeSslWrapper

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import h5py

def save(file_path, all_hidden, all_path, all_latent, all_label, all_mse):
    """Save the representations as h5 file format

    Args:
        file_path (str): path to the file
        all_hidden (list): list of hidden representation for each studied layer
        all_path (list): list of path to audio file used for the experiement
        all_latent (list): list of SAE latent space for each layer under study
        all_label (list): list of label associated with each input data
        all_mse (list): average mse loss for each data of the reconstructed latent spaces
    """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('path', data=all_path, compression='gzip')
        f.create_dataset('hidden', data=all_hidden, compression='gzip')
        f.create_dataset('latent', data=all_latent, compression='gzip')
        f.create_dataset('label', data=all_label, compression='gzip')
        f.create_dataset('mse', data=all_mse, compression='gzip')

def load(file_path):
    with h5py.File(file_path, 'r') as f:
        loaded_hidden = f['hidden'][:]
        loaded_path = f['path'][:]
        loaded_latent = f['latent'][:]
        loaded_label = f['label'][:]
        loaded_mse = f['mse'][:]

    return loaded_hidden, loaded_path, loaded_latent, loaded_label, loaded_mse


def extract(model, dataloader, device, out_file_path):
    """Extract hidden representation and SAE latent space from pretrained audio encoder

    Args:
        model (nn.Module): model including audio encoder and sparse autoencoders
        dataloader (Dataloader): loader for the dataset on which to extract the representations     
        device (str): device on which performing the process        
        out_file_path (str): path where to save the representations

    Returns:
        _type_: _description_
    """
    mse = torch.nn.MSELoss()
    all_mse = []
    model.eval()

    all_hidden = []
    all_latent = []
    all_label = []
    all_path = []

    with torch.no_grad():
        for i, (audio, label, path) in enumerate(tqdm.tqdm(dataloader)):
            # hidden, latent, hidden_hat : lists containing the representation according to a given layer. 
            # the associated layer can be retrieved in cfg["model"]["layer_indices"]
            hidden, latent, hidden_hat = model(audio.to(device))
            n_rep = len(hidden)
            
            mse_avg = 0.0
            for (hh, hh_hat) in zip(hidden, hidden_hat):
                mse_avg+= mse(hh_hat,hh).item()
            
            all_mse.append(mse_avg/n_rep)
            all_hidden.append([h.detach().cpu().numpy() for h in hidden])
            #all_hidden_hat.append([h.detach().cpu().numpy() for h in hidden_hat])
            all_latent.append([l.detach().cpu().numpy() for l in latent])
            all_label.append(label.cpu().numpy())
            all_path.append(path)

        save(file_path=out_file_path, 
             all_hidden=all_hidden,
             all_path=all_path,
             all_latent=all_latent,
             all_label=all_label,
             all_mse=all_mse)
        
        all_mse = np.array(all_mse)
        mse_avg = np.mean(all_mse)
        mse_std = np.std(all_mse)

        return mse_avg, mse_std



if __name__ == "__main__":
    from audIBle.config.sae_ssl_cfg import conf, common_parameters
    from audIBle.config.utils import merge_configs
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_id', type=str, help='Configuration ID for experiment setup')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for experiment reproductibility")
    parser.add_argument("--enc_type",type=str,help="Name of the audio encoder")
    parser.add_argument("--sae_method",type=str,help="Method to obtain asparse latent space")
    parser.add_argument("--sae_dim",type=int,help="Dimension of the sparse latent space")
    parser.add_argument("--sparsity",type=float,help="Sparsity ratio in the SAE latent representation")
    parser.add_argument("--dataset_name",type=str,help="Sparsity ratio in the SAE latent representation")

    args = parser.parse_args()

    exp_conf = conf[args.conf_id]
    config = merge_configs(common_parameters, exp_conf)
    exp_root = os.path.join(os.environ["EXP_ROOT"],"train/SAE/ssl/")
    exp_name = f"{args.conf_id}_{args.enc_type}_{args.dataset_name}_SAE_{args.sae_method}_{args.sae_dim}_{int(args.sparsity*100)}_seed_{args.seed}"
    exp_path = os.path.join(exp_root, exp_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(os.path.join(exp_path,"config.json"), "r") as fh:
        cfg = json.load(fh)

    model = SaeSslWrapper(**cfg["model"])
    ckpt = torch.load(os.path.join(exp_path,"best_model.pth"),weights_only=True, map_location=device)
    model.load_state_dict(ckpt)

    cfg["data"]["valid"]["return_path"] = True
    dataset = select_dataset(dataset_name=cfg["data"]["dataset_name"], **cfg["data"]["valid"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    layer_indices = cfg["model"]["layer_indices"] + [-1]
    suff = '_'.join(str(ll) for ll in layer_indices)
    out_file_path = os.path.join(exp_path,f'extract_rep_{suff}.h5')
    print(f"Representations are saved in : {out_file_path}")
    extract(model, dataloader, device, out_file_path)







