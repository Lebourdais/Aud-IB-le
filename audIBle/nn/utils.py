import torch
import torch.nn as nn

def freeze_model(model, eval_mode=True):
    """
    Completely freeze a model, including BatchNorm statistics.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to freeze
    eval_mode : bool, default=True
        Whether to put the model in evaluation mode
        
    Returns:
    --------
    model : torch.nn.Module
        The frozen model
    """
    # Step 1: Set all parameters to not require gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Step 2: Set model to evaluation mode to freeze BatchNorm statistics
    if eval_mode:
        model.eval()
    
    # Step 3: Override the BatchNorm layers to ensure they don't update statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False  # Don't track running stats
            module.running_mean = module.running_mean.detach()  # Detach running mean
            module.running_var = module.running_var.detach()    # Detach running variance
    
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


