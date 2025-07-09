import torch
import numpy as np
import os
import tqdm
import time

from audIBle.data.datasets import select_dataset
from audIBle.nn.pretrained_models import AudioClassifier
from audIBle.nn.utils import count_parameters

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json

def plot_att_weights(avg_att_weights):
    fig, ax = plt.subplots(1, 1, figsize=(4, 5), layout="tight")
    n_layer = avg_att_weights.shape[0]
    ax.plot(avg_att_weights, color="k", linewidth=2, marker='o')
    ax.grid()
    ax.set_title(f"Average attention weights over layers")
    ax.set_xlabel("Layer index")
    ax.set_xticks(range(n_layer))
    ax.set_xlim([0,n_layer-1])

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
    exp_name = f"{conf_id}_{config["data"]["dataset_name"]}_{seed}"

    config["conf_id"] = conf_id
    config["seed"] = seed
    exp_dir = os.path.join(config["exp_dir"], exp_name)

    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    writer = SummaryWriter(log_dir=exp_dir)
    # Save the configuration

    # Load the dataset
    import pprint
    pprint.pprint(config["data"])

    train_set = select_dataset(dataset_name=config["data"]["dataset_name"], 
                               **config["data"]["train"])

    valid_set = select_dataset(dataset_name=config["data"]["dataset_name"], 
                               **config["data"]["valid"])
    
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=config["optim"]["batch_size"], shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=config["optim"]["batch_size"], shuffle=False)
    
    # GPU omg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # audio classifier with linear head on top of SSL model
    model = AudioClassifier(**config["model"])
    n_layer_select = len(config["model"]["use_layer_indices"])
    print(f"Number of parameters in the model: {count_parameters(model)*1e-6:.2f}M")
    model.to(device)

    # Define loss function and optimizer
    l_classif = torch.nn.CrossEntropyLoss(reduction="mean")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["optim"]["learning_rate"])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["patience_scheduler"], verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["optim"]["patience_scheduler"], gamma=0.5)
    best_loss = float('inf')

    # Training loop     
    for epoch in range(config["optim"]["epochs"]):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm.tqdm(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs)

            # Compute losses
            loss = l_classif(logits, labels)

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
            val_att_weights = []
            for i, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                if n_layer_select > 1:
                    logits,att_weights = model(inputs,return_att_weights=True)
                else:
                    logits = model(inputs)
                    
                loss = l_classif(logits, labels)
                valid_loss += loss.item()
                if n_layer_select > 1:
                    val_att_weights.append(att_weights.mean(dim=0))
            

            # one scheduler step
            #scheduler.step(valid_loss / len(valid_dataloader))
            scheduler.step()
            len_valid = len(valid_dataloader)
            writer.add_scalar('valid/loss', valid_loss/len_valid, epoch )
            if n_layer_select > 1:
                avg_att_weights = torch.stack(val_att_weights).mean(dim=0).detach().cpu().numpy()
                fig = plot_att_weights(avg_att_weights=avg_att_weights)
                writer.add_figure(f'valid/avg_att_weights', fig, epoch)
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

    # To train on ESC-50
    #from audIBle.config.ssl_downstream_cfg import conf, common_parameters
    # To train on TIMIT
    
    # To train on VocalSet
    
    from audIBle.config.utils import merge_configs
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_id', type=str, help='Configuration ID for experiment setup')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for experiment reproductibility")
    parser.add_argument('--dataset', type=str, help='Name of the dataset under study to select the appropriate configuration file')
    args = parser.parse_args()

    
    if args.dataset == "esc50":
        from audIBle.config.ssl_downstream_esc50 import conf, common_parameters
    elif args.dataset == "timit":
        from audIBle.config.ssl_downstream_timit import conf, common_parameters
    elif args.dataset == "vocalset":
        from audIBle.config.ssl_downstream_vocalset import conf, common_parameters
    else: 
        raise Exception(f"No dataset found, try among ['esc50', 'timit', 'vocalset']")

    import pprint
    print("Common parameters before merge")
    pprint.pprint(common_parameters)

    exp_conf = conf[args.conf_id]
    config = merge_configs(common_parameters, exp_conf)

    print("Config after merge")
    pprint.pprint(config)

    train(config, conf_id=args.conf_id, seed=args.seed)
    



    

    
