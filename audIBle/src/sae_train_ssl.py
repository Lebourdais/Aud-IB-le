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
            for hh, zz, hh_hat in zip(hidden, sae_latent, hidden_reconstruct):
                loss_sae = l_reconstruct(hh_hat, hh)

                # optimize L1 loss only in vanilla SAE
                if is_vanilla_sae: 
                    loss_sparse = torch.norm(zz, p=1)
                    loss = lambda_l2 * loss_sae + lambda_l1 * loss_sparse
                else:
                    loss = loss_sae


            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update learning rate
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + i)

            # logging
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('train/MSE_sae', loss_sae.item(), epoch * len(dataloader) + i)
            if is_vanilla_sae:
                writer.add_scalar('train/L1_sae', loss_sae.item(), epoch * len(dataloader) + i)
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{config["optim"]["epochs"]}], Loss: {running_loss/len(dataloader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_loss_sae = 0.0
            valid_loss_sparse = 0.0
            for i, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                hidden, sae_latent, hidden_reconstruct = model(inputs)

                # Compute losses
                for hh, zz, hh_hat in zip(hidden, sae_latent, hidden_reconstruct):
                    loss_sae = l_reconstruct(hh_hat, hh)

                    # optimize L1 loss only in vanilla SAE
                    if is_vanilla_sae: 
                        loss_sparse = torch.norm(zz, p=1)
                        loss = lambda_l2 * loss_sae + lambda_l1 * loss_sparse
                    else:
                        loss = loss_sae


                valid_loss += loss.item()
                valid_loss_sae += loss_sae.item()

            # one scheduler step
            #scheduler.step(valid_loss / len(valid_dataloader))
            scheduler.step()
            len_valid = len(valid_dataloader)
            writer.add_scalar('valid/loss', valid_loss/len_valid, epoch )
            writer.add_scalar('valid/MSE_sae', valid_loss_sae/len_valid, epoch )
            if is_vanilla_sae:
                writer.add_scalar('valid/L1_sae', valid_loss_sparse/len_valid, epoch )
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

def trainddp(config, conf_id, seed, rank, world_size):
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
    dataloader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=config["optim"]["batch_size"], 
        sampler=train_sampler,
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=config["optim"]["batch_size"], 
        sampler=valid_sampler,
        num_workers=4,
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
            for hh, zz, hh_hat in zip(hidden, sae_latent, hidden_reconstruct):
                loss_sae = l_reconstruct(hh_hat, hh)

                # optimize L1 loss only in vanilla SAE
                if is_vanilla_sae: 
                    loss_sparse = torch.norm(zz, p=1)
                    loss = lambda_l2 * loss_sae + lambda_l1 * loss_sparse
                else:
                    loss = loss_sae

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Only log on rank 0
            if rank == 0:
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + i)
                writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('train/MSE_sae', loss_sae.item(), epoch * len(dataloader) + i)
                if is_vanilla_sae:
                    writer.add_scalar('train/L1_sae', loss_sparse.item(), epoch * len(dataloader) + i)
        
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
            valid_loss_sae = 0.0
            valid_loss_sparse = 0.0
            
            for i, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Forward pass
                hidden, sae_latent, hidden_reconstruct = model(inputs)

                # Compute losses
                for hh, zz, hh_hat in zip(hidden, sae_latent, hidden_reconstruct):
                    loss_sae = l_reconstruct(hh_hat, hh)

                    # optimize L1 loss only in vanilla SAE
                    if is_vanilla_sae: 
                        loss_sparse = torch.norm(zz, p=1)
                        loss = lambda_l2 * loss_sae + lambda_l1 * loss_sparse
                    else:
                        loss = loss_sae


                valid_loss += loss.item()
                valid_loss_sae += loss_sae.item()

            # Average validation losses across all processes
            len_valid = len(valid_dataloader)
            valid_loss_tensor = torch.tensor(valid_loss / len_valid, device=device)
            valid_loss_sae_tensor = torch.tensor(valid_loss_sae / len_valid, device=device)
            
            dist.all_reduce(valid_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(valid_loss_sae_tensor, op=dist.ReduceOp.SUM)
            
            avg_valid_loss = valid_loss_tensor.item() / world_size
            avg_valid_loss_sae = valid_loss_sae_tensor.item() / world_size
            
            if is_vanilla_sae:
                valid_loss_sparse_tensor = torch.tensor(valid_loss_sparse / len_valid, device=device)
                dist.all_reduce(valid_loss_sparse_tensor, op=dist.ReduceOp.SUM)
                avg_valid_loss_sparse = valid_loss_sparse_tensor.item() / world_size

            scheduler.step()
            
            # Only log and save on rank 0
            if rank == 0:
                writer.add_scalar('valid/loss', avg_valid_loss, epoch)
                writer.add_scalar('valid/MSE_sae', avg_valid_loss_sae, epoch)
                if is_vanilla_sae:
                    writer.add_scalar('valid/L1_sae', avg_valid_loss_sparse, epoch)
                print(f"Validation Loss: {avg_valid_loss:.4f}")

                # Save the model if the validation loss is the best so far
                if avg_valid_loss <= best_loss:
                    best_loss = avg_valid_loss
                    # Save the model without DDP wrapper
                    torch.save(model.module.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
                    print(f"Model saved at epoch {epoch+1} with validation loss: {avg_valid_loss:.4f}")

    if rank == 0:
        writer.close()
    
    cleanup()

def main_worker(rank, world_size, config, conf_id, seed):
    """Main worker function for each process."""
    trainddp(config, conf_id, seed, rank, world_size)

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


    
    



    

    
