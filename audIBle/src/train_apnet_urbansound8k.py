import torch
import numpy as np
import os
import tqdm
import time

from audIBle.data.datasets import UrbanSound8k
from audIBle.src.apnet import APNet
from audIBle.src.losses import ProtoLoss

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json


def plot_spec_in_out(spec, spec_hat, epoch):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
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

    return fig

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def default_config():
    return {
        'data_dir': '/lium/corpus/vrac/tmario/sed/urbansound8k/urbansound8k',
        'exp_dir': '/lium/raid-b/tmario/train/proto/apnet_urbansound8k',
        'epochs': 200,
        'batch_size': 256,
        'learning_rate': 0.001,
        'loss_factors': (1, 0.5, 0.01), # loss = alpha*loss_class + beta*loss_reconstruct + gamma*loss_proto
        'patience_early_stop': 15,
        'patience_scheduler': 200,
        'folds_train': [1,2,3,4,5,6,7,8],
        'folds_valid': [9],
        'model': {
            'n_classes': 10,
            'n_filters': [8, 16, 32],
            'filter_size': 3,
            'pool_size': 2,
            'seg_len': 4, 
            'dilation_rate': (1, 1),
            'use_batch_norm': False,
            'n_prototypes': 50,
            'distance': 'euclidean',
            'use_weighted_sum': True,
            'mel_spec_param': {"sample_rate":22050, "n_fft":4096, "hop_length":1024, "n_mels":128}
        }
    }


def train(config):

    fix_seed(42)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{timestamp}_{config["epochs"]}epochs_{config["batch_size"]}batchsize_{config["learning_rate"]}lr"

    exp_dir = os.path.join(config["exp_dir"], exp_name)

    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    writer = SummaryWriter(log_dir=exp_dir)
    # Save the configuration

    # Load the dataset
    #dataset = ESC_50(root=config["data_dir"], part='train')
    dataset_root = config["data_dir"]
    dataset = UrbanSound8k(csv_path=os.path.join(dataset_root, "metadata/UrbanSound8K.csv"),
                           audio_dir=os.path.join(dataset_root, "audio"),
                           sample_rate=config["model"]["mel_spec_param"]["sample_rate"],
                           folds_to_use=config["folds_train"],
                           duration=4.0,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    valid_dataset = UrbanSound8k(csv_path=os.path.join(dataset_root, "metadata/UrbanSound8K.csv"),
                           audio_dir=os.path.join(dataset_root, "audio"),
                           sample_rate=config["model"]["mel_spec_param"]["sample_rate"],
                           folds_to_use=config["folds_valid"],
                           duration=4.0,)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # APNet model
    model = APNet(
        **config["model"]
    )
    print(f"Number of parameters in the model: {count_parameters(model)*1e-6:.2f}M")

    # GPU omg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    l_class = torch.nn.CrossEntropyLoss(reduction='mean')
    l_reconstruct = torch.nn.MSELoss(reduction='mean')
    l_proto = ProtoLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["patience_scheduler"], verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["patience_scheduler"], gamma=0.5)
    best_loss = float('inf')
    patience_early_stop = config["patience_early_stop"]

    idx_visu = [1,3,6]

    # Training loop     
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm.tqdm(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            pred, x_hat, dist_full, spec_feat, latent = model(inputs, return_all=True)

            # Compute losses
            loss_class = l_class(pred, labels)
            loss_reconstruct = l_reconstruct(x_hat, spec_feat)
            loss_proto = l_proto(Z=latent, P=model.get_prototypes())

            # Total loss
            alpha, beta, gamma = config["loss_factors"]
            loss = alpha*loss_class + beta*loss_reconstruct + gamma*loss_proto
            #loss = loss_reconstruct
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update learning rate
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + i)

            # logging
            writer.add_scalar('train/total', loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('train/class', loss_class.item(), epoch * len(dataloader) + i)
            writer.add_scalar('train/reconstruct', loss_reconstruct.item(), epoch * len(dataloader) + i)
            writer.add_scalar('train/proto', loss_proto.item(), epoch * len(dataloader) + i)

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config["epochs"]}], Loss: {running_loss/len(dataloader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for i, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                pred, x_hat, dist_full, spec_feat, latent = model(inputs, return_all=True)

                loss_class = l_class(pred, labels)
                loss_reconstruct = l_reconstruct(x_hat, spec_feat)
                loss_proto = l_proto(Z=latent, P=model.get_prototypes())

                loss = alpha*loss_class + beta*loss_reconstruct + gamma*loss_proto
                #loss = loss_reconstruct

                if i in idx_visu:
                    fig = plot_spec_in_out(spec_feat.detach(), x_hat.detach(), epoch)
                    writer.add_figure(f'train/spec_in_out{i}', fig, epoch)

                valid_loss += loss.item()
            # one scheduler step
            #scheduler.step(valid_loss / len(valid_dataloader))
            scheduler.step()

            writer.add_scalar('valid/total', valid_loss / len(valid_dataloader), epoch)
            writer.add_scalar('valid/class', loss_class.item(), epoch)
            writer.add_scalar('valid/reconstruct', loss_reconstruct.item(), epoch)
            writer.add_scalar('valid/proto', loss_proto.item(), epoch)
            print(f"Validation Loss: {valid_loss / len(valid_dataloader):.4f}")

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
    config = default_config()
    train(config)
    print("Training completed.")
    print("Model saved in:", config["exp_dir"])
    print("Tensorboard logs saved in:", config["exp_dir"])
    print("Best model saved as best_model.pth")



    

    