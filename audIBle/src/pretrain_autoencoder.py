import torch
import numpy as np
import os
import tqdm
import time

from audIBle.data.datasets import UrbanSound8k
from audIBle.nn.autoencoders import SpecAE
from audIBle.nn.utils import count_parameters

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

def train(config, conf_id, seed):

    fix_seed(seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_id = config["job_id"]
    # exp_name = f"{conf_id}_{timestamp}_{job_id}_spec_autoencoder_urbasound8k_{seed}"
    exp_name = f"{conf_id}_spec_autoencoder_urbasound8k_{seed}"

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
    
    # Autoencoder model
    model = SpecAE(**config["model"])
    print(f"Number of parameters in the model: {count_parameters(model)*1e-6:.2f}M")

    # GPU omg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    l_reconstruct = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=config["optim"]["learning_rate"])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["patience_scheduler"], verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["optim"]["patience_scheduler"], gamma=0.5)
    best_loss = float('inf')
    patience_early_stop = config["optim"]["patience_early_stop"]

    idx_visu = [1,3,6]

    # Training loop     
    for epoch in range(config["optim"]["epochs"]):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm.tqdm(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            x_hat, x, z, z_enh = model(inputs)

            # Compute losses
            loss = l_reconstruct(x_hat, x)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update learning rate
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + i)

            # logging
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + i)

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{config["optim"]["epochs"]}], Loss: {running_loss/len(dataloader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for i, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                x_hat, x, z, z_enh = model(inputs)

                loss = l_reconstruct(x_hat, x)

                if i in idx_visu:
                    fig = plot_spec_in_out(x.detach().squeeze(), x_hat.detach().squeeze(), epoch)
                    writer.add_figure(f'train/spec_in_out{i}', fig, epoch)

                valid_loss += loss.item()
            # one scheduler step
            #scheduler.step(valid_loss / len(valid_dataloader))
            scheduler.step()
            writer.add_scalar('valid/loss', loss.item(), epoch)
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

    from audIBle.config.autoenc_cfg import conf, common_parameters
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
    



    

    