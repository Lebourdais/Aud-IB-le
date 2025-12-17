import numpy as np

def disentanglement_score(reg_w: np.array, eps=1e-6):
    reg_w = np.abs(reg_w)
    n_codes = reg_w.shape[0]
    n_factors = reg_w.shape[1]
    probs = reg_w / (np.sum(reg_w, axis=1, keepdims=True) + eps)
    # entropy
    H = -np.sum(probs * np.log(probs + eps)/np.log(n_factors), axis=1, keepdims=True)
    # disentanglement score
    D = 1 - H

    # weighted average with importance weighting 
    code_importance = np.sum(reg_w,axis=1, keepdims=True) / (np.sum(reg_w, keepdims=True) + eps)
    D_avg = np.mean(code_importance * D)

    return D, code_importance, D_avg

def completeness_score(reg_w: np.array, eps=1e-6):
    reg_w = np.abs(reg_w)
    n_codes = reg_w.shape[0]
    n_factors = reg_w.shape[1]
    probs_tilde = reg_w / (np.sum(reg_w, axis=0, keepdims=True) + eps)
    # entropy
    H = -np.sum(probs_tilde * np.log(probs_tilde + eps)/np.log(n_codes), axis=0, keepdims=True)
    # completeness score
    C = 1 - H

    return C