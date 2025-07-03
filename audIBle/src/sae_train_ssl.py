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

    dataset_name = config["data"]["dataset_name"]
    model_name = config["model"]["encoder_type"]
    sae_type = config["model"]["sae_method"]
    sae_dim = config["model"]["sae_dim"]
    sparsity = config["model"]["sparsity"]
    layer_indices = config["model"]["layer_indices"] + [-1]
    exp_name = f"{conf_id}_{model_name}_{dataset_name}_SAE_{sae_type}_{sae_dim}_{int(sparsity*100)}_seed_{seed}"

    config["conf_id"] = conf_id
    config["seed"] = seed
    exp_dir = os.path.join(config["exp_dir"], exp_name)

    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    writer = SummaryWriter(log_dir=exp_dir)
    # Save the configuration
    train_set = select_dataset(dataset_name=config["data"]["dataset_name"], 
                               **config["data"]["train"])

    valid_set = select_dataset(dataset_name=config["data"]["dataset_name"], 
                               **config["data"]["valid"])
    
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=config["optim"]["batch_size"], shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=config["optim"]["batch_size"], shuffle=False)
    
    # GPU omg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SaeSslWrapper(**config["model"])

    is_vanilla_sae = config["model"]["sae_method"] == "vanilla"

    print(f"Number of parameters in the model: {count_parameters(model)*1e-6:.2f}M")
    model.to(device)

    # Define loss function and optimizer
    l_reconstruct = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=config["optim"]["learning_rate"])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["patience_scheduler"], verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["optim"]["patience_scheduler"], gamma=0.5)
    best_loss = float('inf')
    patience_early_stop = config["optim"]["patience_early_stop"]

    lambda_l2, lambda_l1 = config["optim"]["loss_weights"]

    # Training loop     
    for epoch in range(config["optim"]["epochs"]):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm.tqdm(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            hidden, sae_latent, hidden_reconstruct = model(inputs)

            # Compute losses
            mse_layers = []
            l1_layers = []
            for l_idx, (hh, zz, hh_hat) in enumerate(zip(hidden, sae_latent, hidden_reconstruct)):
                loss_sae = l_reconstruct(hh_hat, hh)
                mse_layers.append(loss_sae)
                # optimize L1 loss only in vanilla SAE
                if is_vanilla_sae: 
                    loss_sparse = torch.norm(zz, p=1)
                    #loss = lambda_l2 * loss_sae + lambda_l1 * loss_sparse
                    l1_layers.append(loss_sparse)
                    writer.add_scalar(f'train/L1_layer_{layer_indices[l_idx]}', loss_sae.item(), epoch * len(dataloader) + i)
                else:
                    loss = loss_sae

                writer.add_scalar(f'train/MSE_layer_{layer_indices[l_idx]}', loss_sae.item(), epoch * len(dataloader) + i)

            mse_loss_avg = torch.stack(mse_layers).mean(dim=0)
            if is_vanilla_sae:
                sparse_loss_avg = torch.stack(l1_layers).mean(dim=0)
                loss = lambda_l2 * mse_loss_avg + lambda_l1 * sparse_loss_avg
            else:
                loss = mse_loss_avg                

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update learning rate
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + i)

            # logging
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('train/Avg_MSE_sae', mse_loss_avg.item(), epoch * len(dataloader) + i)
            if is_vanilla_sae:
                writer.add_scalar('train/L1_sae', sparse_loss_avg.item(), epoch * len(dataloader) + i)
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{config["optim"]["epochs"]}], Loss: {running_loss/len(dataloader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for i, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                hidden, sae_latent, hidden_reconstruct = model(inputs)

                mse_layers = [0.0 for _ in range(len(layer_indices))]
                l1_layers = [0.0 for _ in range(len(layer_indices))]
                for l_idx, (hh, zz, hh_hat) in enumerate(zip(hidden, sae_latent, hidden_reconstruct)):
                    loss_sae = l_reconstruct(hh_hat, hh)
                    mse_layers[l_idx] += loss_sae.item()

                    # optimize L1 loss only in vanilla SAE
                    if is_vanilla_sae: 
                        loss_sparse = torch.norm(zz, p=1)
                        #loss = lambda_l2 * loss_sae + lambda_l1 * loss_sparse
                        l1_layers[l_idx] += loss_sparse.item()
            
            len_valid = len(valid_dataloader)
            # average MSE/L1 over all SAE
            val_mse_avg = torch.Tensor(mse_layers).mean()
            if is_vanilla_sae:
                val_l1_avg = torch.Tensor(l1_layers).mean()
                valid_loss = lambda_l2 * val_mse_avg + lambda_l1 * val_l1_avg
                writer.add_scalar('valid/L1_sae', val_l1_avg/len_valid, epoch )
            else:
                valid_loss = val_mse_avg

            # log average loss for each layer across all the valid set

            for l_idx in range(len(layer_indices)):
                writer.add_scalar(f'valid/MSE_layer_{layer_indices[l_idx]}', mse_layers[l_idx]/len_valid, epoch)
                if is_vanilla_sae:
                    writer.add_scalar(f'valid/L1_layer_{layer_indices[l_idx]}', l1_layers[l_idx]/len_valid, epoch)


            
            writer.add_scalar('valid/loss', valid_loss/len_valid, epoch )
            writer.add_scalar('valid/MSE_sae', val_mse_avg/len_valid, epoch )
            # one scheduler step
            #scheduler.step(valid_loss / len(valid_dataloader))
            scheduler.step()

            print(f"Validation Loss: {valid_loss / len_valid:.4f}")

        # Save the model if the validation loss is the best so far
        if valid_loss <= best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            print(f"Model saved at epoch {epoch+1} with validation loss: {valid_loss / len(valid_dataloader):.4f}")

    writer.close()


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_ddp(config, conf_id, seed, rank, world_size):
    # Setup distributed training
    setup(rank, world_size)
    
    fix_seed(seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_id = config["job_id"]

    dataset_name = config["data"]["dataset_name"]
    model_name = config["model"]["encoder_type"]
    sae_type = config["model"]["sae_method"]
    sae_dim = config["model"]["sae_dim"]
    sparsity = config["model"]["sparsity"]
    layer_indices = config["model"]["layer_indices"] + [-1]
    exp_name = f"{conf_id}_{model_name}_{dataset_name}_SAE_{sae_type}_{sae_dim}_{int(sparsity*100)}_seed_{seed}"

    config["conf_id"] = conf_id
    config["seed"] = seed
    exp_dir = os.path.join(config["exp_dir"], exp_name)

    # Only create directories and save config on rank 0
    if rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        writer = SummaryWriter(log_dir=exp_dir)

    # Load datasets
    train_set = select_dataset(dataset_name=config["data"]["dataset_name"], 
                               **config["data"]["train"])
    valid_set = select_dataset(dataset_name=config["data"]["dataset_name"], 
                               **config["data"]["valid"])
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create data loaders with distributed samplers
    # Set num_workers=0 to avoid multiprocessing issues with DDP
    dataloader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=config["optim"]["batch_size"], 
        sampler=train_sampler,
        num_workers=0, 
        pin_memory=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=config["optim"]["batch_size"], 
        sampler=valid_sampler,
        num_workers=0, 
        pin_memory=True
    )
    
    # Initialize model on the specific GPU
    device = torch.device(f"cuda:{rank}")
    model = SaeSslWrapper(**config["model"])
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    is_vanilla_sae = config["model"]["sae_method"] == "vanilla"

    if rank == 0:
        print(f"Number of parameters in the model: {count_parameters(model)*1e-6:.2f}M")

    # Define loss function and optimizer
    l_reconstruct = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optim"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["optim"]["patience_scheduler"], gamma=0.5)
    
    best_loss = float('inf')
    patience_early_stop = config["optim"]["patience_early_stop"]
    lambda_l2, lambda_l1 = config["optim"]["loss_weights"]

    # Training loop     
    for epoch in range(config["optim"]["epochs"]):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        
        # Only show progress bar on rank 0
        dataloader_iter = tqdm.tqdm(dataloader) if rank == 0 else dataloader
        
        for i, (inputs, labels) in enumerate(dataloader_iter):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            hidden, sae_latent, hidden_reconstruct = model(inputs)

            # Compute losses
            mse_layers = []
            l1_layers = []
            for l_idx, (hh, zz, hh_hat) in enumerate(zip(hidden, sae_latent, hidden_reconstruct)):
                loss_sae = l_reconstruct(hh_hat, hh)
                mse_layers.append(loss_sae)
                # optimize L1 loss only in vanilla SAE
                if is_vanilla_sae: 
                    loss_sparse = torch.norm(zz, p=1)
                    l1_layers.append(loss_sparse)
                    if rank == 0:
                        writer.add_scalar(f'train/L1_layer_{layer_indices[l_idx]}', loss_sparse.item(), epoch * len(dataloader) + i)
                else:
                    loss = loss_sae

                if rank == 0:
                    writer.add_scalar(f'train/MSE_layer_{layer_indices[l_idx]}', loss_sae.item(), epoch * len(dataloader) + i)

            mse_loss_avg = torch.stack(mse_layers).mean(dim=0)
            if is_vanilla_sae:
                sparse_loss_avg = torch.stack(l1_layers).mean(dim=0)
                loss = lambda_l2 * mse_loss_avg + lambda_l1 * sparse_loss_avg
            else:
                loss = mse_loss_avg

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Only log on rank 0
            if rank == 0:
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + i)
                writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('train/Avg_MSE_sae', mse_loss_avg.item(), epoch * len(dataloader) + i)
                if is_vanilla_sae:
                    writer.add_scalar('train/L1_sae', sparse_loss_avg.item(), epoch * len(dataloader) + i)
        
        # Average loss across all processes
        running_loss_tensor = torch.tensor(running_loss / len(dataloader), device=device)
        dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
        avg_running_loss = running_loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{config['optim']['epochs']}], Loss: {avg_running_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            mse_layers_total = [0.0 for _ in range(len(layer_indices))]
            l1_layers_total = [0.0 for _ in range(len(layer_indices))]
            
            for i, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Forward pass
                hidden, sae_latent, hidden_reconstruct = model(inputs)

                mse_layers = [0.0 for _ in range(len(layer_indices))]
                l1_layers = [0.0 for _ in range(len(layer_indices))]
                for l_idx, (hh, zz, hh_hat) in enumerate(zip(hidden, sae_latent, hidden_reconstruct)):
                    loss_sae = l_reconstruct(hh_hat, hh)
                    mse_layers[l_idx] += loss_sae.item()
                    mse_layers_total[l_idx] += loss_sae.item()

                    # optimize L1 loss only in vanilla SAE
                    if is_vanilla_sae: 
                        loss_sparse = torch.norm(zz, p=1)
                        l1_layers[l_idx] += loss_sparse.item()
                        l1_layers_total[l_idx] += loss_sparse.item()

            # Convert to tensors and reduce across all processes
            len_valid = len(valid_dataloader)
            
            # Aggregate layer losses across all processes
            mse_layers_tensor = torch.tensor(mse_layers_total, device=device)
            dist.all_reduce(mse_layers_tensor, op=dist.ReduceOp.SUM)
            mse_layers_avg = mse_layers_tensor / world_size
            
            if is_vanilla_sae:
                l1_layers_tensor = torch.tensor(l1_layers_total, device=device)
                dist.all_reduce(l1_layers_tensor, op=dist.ReduceOp.SUM)
                l1_layers_avg = l1_layers_tensor / world_size

            # Calculate average losses
            val_mse_avg = mse_layers_avg.mean()
            if is_vanilla_sae:
                val_l1_avg = l1_layers_avg.mean()
                valid_loss = lambda_l2 * val_mse_avg + lambda_l1 * val_l1_avg
            else:
                valid_loss = val_mse_avg

            # All reduce the total validation loss
            valid_loss_tensor = torch.tensor(valid_loss, device=device)
            dist.all_reduce(valid_loss_tensor, op=dist.ReduceOp.SUM)
            avg_valid_loss = valid_loss_tensor.item() / world_size

            scheduler.step()
            
            # Only log and save on rank 0
            if rank == 0:
                # Log average loss for each layer
                for l_idx in range(len(layer_indices)):
                    writer.add_scalar(f'valid/MSE_layer_{layer_indices[l_idx]}', mse_layers_avg[l_idx]/len_valid, epoch)
                    if is_vanilla_sae:
                        writer.add_scalar(f'valid/L1_layer_{layer_indices[l_idx]}', l1_layers_avg[l_idx]/len_valid, epoch)

                writer.add_scalar('valid/loss', avg_valid_loss/len_valid, epoch)
                writer.add_scalar('valid/MSE_sae', val_mse_avg/len_valid, epoch)
                if is_vanilla_sae:
                    writer.add_scalar('valid/L1_sae', val_l1_avg/len_valid, epoch)
                    
                print(f"Validation Loss: {avg_valid_loss / len_valid:.4f}")

                # Save the model if the validation loss is the best so far
                if avg_valid_loss <= best_loss:
                    best_loss = avg_valid_loss
                    # Save the model without DDP wrapper
                    torch.save(model.module.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
                    print(f"Model saved at epoch {epoch+1} with validation loss: {avg_valid_loss / len_valid:.4f}")

    if rank == 0:
        writer.close()
    
    cleanup()

def main_worker(rank, world_size, config, conf_id, seed):
    """Main worker function for each process."""
    train_ddp(config, conf_id, seed, rank, world_size)

def main(config, conf_id, seed):
    """Main function to spawn multiple processes."""
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    
    # Spawn processes
    mp.spawn(
        main_worker,
        args=(world_size, config, conf_id, seed),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":

    from audIBle.config.sae_ssl_cfg import conf, common_parameters
    from audIBle.config.utils import merge_configs
    import argparse
    import torch 

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_id', type=str, help='Configuration ID for experiment setup')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for experiment reproductibility")
    args = parser.parse_args()

    exp_conf = conf[args.conf_id]
    config = merge_configs(common_parameters, exp_conf)

    import pprint
    pprint.pprint(config)

    if torch.cuda.device_count() > 1:
        mp.set_start_method('spawn', force=True)
        main(config, args.conf_id, args.seed)
    else:
        train(config, conf_id=args.conf_id, seed=args.seed)


    
    



    

    
