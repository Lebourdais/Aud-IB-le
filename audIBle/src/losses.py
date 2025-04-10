import torch

class ProtoLoss():
    def __init__(self):
        pass

    def __call__(self, P, Z):
        # If the tensor is 3D, take the mean along the last axis
        P = P.flatten(start_dim=1)
        Z = Z.flatten(start_dim=1)
        dist_mat = torch.cdist(P, Z, p=2)
        
        # Calculate error_1: mean of the minimum values along the first axis (axis=0)
        error_1 = dist_mat.min(dim=0)[0].mean()

        # Calculate error_2: mean of the minimum values along the second axis (axis=1)
        error_2 = dist_mat.min(dim=1)[0].mean()

        return error_1 + error_2

    

