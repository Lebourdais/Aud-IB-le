import torch
import numpy as np
import os
import tqdm
import time

from audIBle.data.datasets import UrbanSound8k
from audIBle.nn.autoencoders import SpecAE
from audIBle.nn.sparse_classif import SparseClassifier
from audIBle.nn.utils import count_parameters

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json



def plot_pred(spec, spec_hat, scores, idx_class,class_labels):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout="tight")
    vmin = min(spec.min().item(), spec_hat.min().item())
    vmax = max(spec.max().item(), spec_hat.max().item())
    ax[0].imshow(spec.cpu().numpy()[0], aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax[0].set_title('Input Spectrogram')
    cbar = fig.colorbar(ax[0].images[0], ax=ax[0], orientation='vertical')
    cbar.set_label('Amplitude')
    ax[1].imshow(spec_hat.cpu().numpy()[0], aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax[1].set_title('Reconstructed Spectrogram')
    cbar = fig.colorbar(ax[1].images[0], ax=ax[1], orientation='vertical')
    cbar.set_label('Amplitude')

    score = scores.cpu().numpy()[0]
    ax[2].plot(score, color="k", linewidth=2, label="Softmax scores")
    idx = idx_class.cpu().numpy()[0]
    ax[2].vlines(idx, 0, score[idx], color="red", linewidth=2, linestyle="--", label="Target class score")
    ax[2].grid()
    ax[2].set_title(f"Scores for target class {class_labels[idx]}")
    ax[2].set_xticks(range(len(class_labels)))
    ax[2].set_xticklabels(class_labels, rotation=45, ha='right')
    ax[2].set_xlim([0,len(class_labels)-1])

    return fig

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def train(config, conf_id, seed):

    fix_seed(seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_id = config["job_id"]
    # exp_name = f"{conf_id}_{timestamp}_{job_id}_sparse_classif_urbasound8k_{seed}"
    exp_name = f"{conf_id}_sparse_classif_urbasound8k_{seed}"

    config["conf_id"] = conf_id
    config["seed"] = seed
    exp_dir = os.path.join(config["exp_dir"], exp_name)

    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    writer = SummaryWriter(log_dir=exp_dir)
    # Save the configuration

    # Load the dataset
    #dataset = ESC_50(root=config["data_dir"], part='train')
    dataset_root = config["data"]["root"]
    dataset = UrbanSound8k(csv_path=os.path.join(dataset_root, "metadata/UrbanSound8K.csv"),
                           audio_dir=os.path.join(dataset_root, "audio"),
                           sample_rate=config["sample_rate"],
                           folds_to_use=config["data"]["folds_train"],
                           duration=4.0,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["optim"]["batch_size"], shuffle=True)

    valid_dataset = UrbanSound8k(csv_path=os.path.join(dataset_root, "metadata/UrbanSound8K.csv"),
                           audio_dir=os.path.join(dataset_root, "audio"),
                           sample_rate=config["sample_rate"],
                           folds_to_use=config["data"]["folds_valid"],
                           duration=4.0,)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["optim"]["batch_size"], shuffle=False)
    
    # GPU omg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load autoencoder (pretrained or not)
    autoencoder = SpecAE(**config["model"]["autoencoder"])
    ae_ckpt_path = config["model"]["ae_ckpt_path"]
    if ae_ckpt_path is not None:
        ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
        autoencoder.load_state_dict(ae_ckpt)

    # prepare the sparse classifier
    classif_params = config["model"]["classifier"]
    classif_params["autoencoder"] = autoencoder
    model = SparseClassifier(**classif_params)

    print(f"Number of parameters in the model: {count_parameters(model)*1e-6:.2f}M")
    model.to(device)

    # Define loss function and optimizer
    l_classif = torch.nn.CrossEntropyLoss(reduction="mean")
    l_reconstruct = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=config["optim"]["learning_rate"])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["patience_scheduler"], verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["optim"]["patience_scheduler"], gamma=0.5)
    best_loss = float('inf')
    patience_early_stop = config["optim"]["patience_early_stop"]

    lambda_ce, lambda_sae, lambda_spec = config["optim"]["loss_weights"]

    idx_visu = [1,3,6]

    # Training loop     
    for epoch in range(config["optim"]["epochs"]):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm.tqdm(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            y_hat, spec_reconstruct, spec, hidden, hidden_reconstruct= model(inputs)

            # Compute losses
            loss_spec = l_reconstruct(spec_reconstruct, spec)
            loss_sae = l_reconstruct(hidden_reconstruct, hidden)
            loss_ce = l_classif(y_hat, labels)

            loss = lambda_ce * loss_ce + lambda_sae * loss_sae + lambda_spec * loss_spec

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update learning rate
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + i)

            # logging
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('train/BCE', loss_ce.item(), epoch * len(dataloader) + i)
            writer.add_scalar('train/MSE_sae', loss_sae.item(), epoch * len(dataloader) + i)
            if lambda_spec > 0:
                writer.add_scalar('train/MSE_spec', loss_spec.item(), epoch * len(dataloader) + i)
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{config["optim"]["epochs"]}], Loss: {running_loss/len(dataloader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_loss_ce = 0.0
            valid_loss_sae = 0.0
            valid_loss_spec = 0.0
            for i, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                y_hat, spec_reconstruct, spec, hidden, hidden_reconstruct= model(inputs)

                # Compute losses
                loss_spec = l_reconstruct(spec_reconstruct, spec)
                loss_sae = l_reconstruct(hidden_reconstruct, hidden)
                loss_ce = l_classif(y_hat, labels)

                loss = lambda_ce * loss_ce + lambda_sae * loss_sae + lambda_spec * loss_spec

                if i in idx_visu:
                    scores = torch.nn.functional.softmax(y_hat, dim=-1)
                    fig = plot_pred(spec.detach().squeeze(), spec_reconstruct.detach().squeeze(), scores=scores.detach(), idx_class=labels.detach(), class_labels=valid_dataset.classes)
                    writer.add_figure(f'train/spec_in_out{i}', fig, epoch)

                valid_loss += loss.item()
                valid_loss_ce += loss_ce.item()
                valid_loss_sae += loss_sae.item()
                valid_loss_spec += loss_spec.item()

            # one scheduler step
            #scheduler.step(valid_loss / len(valid_dataloader))
            scheduler.step()
            len_valid = len(valid_dataloader)
            writer.add_scalar('valid/loss', valid_loss/len_valid, epoch )
            writer.add_scalar('valid/BCE', valid_loss_ce/len_valid, epoch )
            writer.add_scalar('valid/MSE_sae', valid_loss_sae/len_valid, epoch )
            if lambda_spec > 0:
                writer.add_scalar('valid/MSE_spec', valid_loss_spec/len_valid, epoch )
            print(f"Validation Loss: {valid_loss / len_valid:.4f}")

        # Save the model if the validation loss is the best so far
        if valid_loss <= best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            print(f"Model saved at epoch {epoch+1} with validation loss: {valid_loss / len(valid_dataloader):.4f}")
        # else:
        #     patience_early_stop -= 1
        #     if patience_early_stop == 0:
        #         print("Early stopping triggered")
        #         break
    writer.close()
    

if __name__ == "__main__":

    from audIBle.config.sparse_classif_cfg import conf, common_parameters
    from audIBle.config.utils import merge_configs
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_id', type=str, help='Configuration ID for experiment setup')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for experiment reproductibility")
    args = parser.parse_args()

    exp_conf = conf[args.conf_id]
    config = merge_configs(common_parameters, exp_conf)

    import pprint
    pprint.pprint(config)

    train(config, conf_id=args.conf_id, seed=args.seed)
    



    

    