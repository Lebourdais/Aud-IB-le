import os
import json
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import tqdm

exp_root = "/lium/raid-b/tmario/train/SAE/ssl"
exp_list = os.listdir(exp_root)
conf_id = "005"
exp_name = [exp for exp in exp_list if exp.startswith(conf_id)][0]
exp_path = os.path.join(exp_root, exp_name)
print(f"{exp_name=}")

# load config for meta informations
with open(os.path.join(exp_path, "config.json"),"r") as ff:
    cfg = json.load(ff)

model = cfg["model"]["encoder_type"]
sparsity = cfg["model"]["sparsity"]
sae_dim = cfg["model"]["sae_dim"]
layer_indices = cfg["model"]["layer_indices"]
print(f"{layer_indices=}")

files = os.listdir(exp_path)
rep_file = os.path.join(exp_path,[f for f in files if f.startswith("extract_rep")][0])

sae_rep = {}
with h5py.File(rep_file, "r") as fh:
    for k in fh.keys():
        data = fh[k][()]
        sae_rep[k] = data
print(sae_rep.keys())

layer_index = 0
sae_latent = sae_rep["latent"][:,layer_index,0,0,:]


# plot distribution of non-zero dimensions in the SAE latent space
pooled_sae_latent = np.sum(sae_latent, axis=0)
nz_dims = np.nonzero(pooled_sae_latent)[-1]
with PdfPages(f"{exp_name}_latent_histograms.pdf") as pdf:
    for dim in tqdm.tqdm(nz_dims):
        plt.figure()
        plt.hist(sae_latent[:, dim], bins=100)
        plt.title(f"Histogram of SAE Latent Activations (Dim {dim})")
        plt.xlabel("Activation Value")
        plt.ylabel("Count")
        plt.tight_layout()
        pdf.savefig()
        plt.close()