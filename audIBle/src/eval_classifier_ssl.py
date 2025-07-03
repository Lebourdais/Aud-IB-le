from audIBle.nn.pretrained_models import AudioClassifier
from audIBle.data.datasets import select_dataset

import torch
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, log_loss

import tqdm

def eval(model: torch.nn.Module, 
         dataset: torch.utils.data.Dataset, 
         metrics: list =["acc", "auc", "f1", "precision", "recall", "log_loss"], 
         device:str=None,):
    """Evaluate the sparse classifier model 

    Args:
        model (nn.Module): model to evaluate (it can be a SparseClassifier model instance)
        dataset (torch.utils.data.Dataset): test dataset on which to evaluate the model
        metrics (list, optional): List of metrics to compute. Defaults to ["acc", "auc", "f1", "precision", "recall", "log_loss", 'mse_spec', 'mse_sae'].
        device (str, optional): Device to use for model inference. Defaults to None.

    Returns:
        dict: Dictionnary with the metric for each file of the training set
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    results = {metric: [] for metric in metrics}
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset)),desc="Evaluating classifier..."):
            x, y = dataset[i]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if isinstance(y, np.int64):
                y = torch.Tensor([y])
            x = x.unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            all_labels.append(y)
            all_probs.append(probs.cpu().numpy()[0])
            all_preds.append(pred.cpu().item())


    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    n_classes = all_probs_np.shape[1]

    if "acc" in metrics:
        acc_per_sample = (all_preds_np == all_labels_np).astype(float)
        results["acc"] = acc_per_sample.tolist()
    if "auc" in metrics:
        try:
            auc_per_sample = []
            for i in range(len(all_labels_np)):
                y_true = np.zeros(n_classes)
                y_true[all_labels_np[i]] = 1
                auc = roc_auc_score(y_true, all_probs_np[i], multi_class='ovr')
                auc_per_sample.append(auc)
            results["auc"] = auc_per_sample
        except Exception:
            results["auc"] = [None] * len(all_labels_np)
    if "f1" in metrics:
        f1_per_sample = []
        for i in range(len(all_labels_np)):
            f1 = f1_score([all_labels_np[i]], [all_preds_np[i]], average="macro", zero_division=0)
            f1_per_sample.append(f1)
        results["f1"] = f1_per_sample
    if "precision" in metrics:
        precision_per_sample = []
        for i in range(len(all_labels_np)):
            precision = precision_score([all_labels_np[i]], [all_preds_np[i]], average="macro", zero_division=0)
            precision_per_sample.append(precision)
        results["precision"] = precision_per_sample
    if "recall" in metrics:
        recall_per_sample = []
        for i in range(len(all_labels_np)):
            recall = recall_score([all_labels_np[i]], [all_preds_np[i]], average="macro", zero_division=0)
            recall_per_sample.append(recall)
        results["recall"] = recall_per_sample
    if "log_loss" in metrics:
        log_loss_per_sample = []
        for i in range(len(all_labels_np)):
            y_true = np.zeros(n_classes)
            y_true[all_labels_np[i]] = 1
            try:
                ll = log_loss([y_true], [all_probs_np[i]], labels=list(range(n_classes)))
            except Exception:
                ll = None
            log_loss_per_sample.append(ll)
        results["log_loss"] = log_loss_per_sample

    return results

if __name__ == "__main__":

    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()

    parser.add_argument('--conf_id', type=str, help="Identifier of the configuration to test.")
    parser.add_argument('--exp_tag', type=str, help="Tag of the experiments to select the appropriate folder.")
    parser.add_argument('--dataset_name', type=str, help="Dataset on which testing the model.")
    parser.add_argument("--seed", type=int, help="Seed used to train the model.")
    parser.add_argument("--samplerate", type=int, help="Working samplerate of the model")
    args = parser.parse_args()

    exp_name = f"{args.conf_id}_{args.dataset_name.lower()}_{args.seed}"
    exp_root = os.path.join(os.environ["EXP_ROOT"], f"train/SAE/{args.exp_tag}/",exp_name)

    with open(os.path.join(exp_root, "config.json"), 'r') as fh:
        cfg = json.load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = AudioClassifier(**cfg["model"])

    # load classifier checkpoint
    classif_ckpt_path = os.path.join(exp_root,"best_model.pth")
    ckpt = torch.load(classif_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.to(device)

    # Prepare the test dataset
    dataset_name = args.dataset_name
    if dataset_name.upper() == "ESC50":
        cfg_data = {"root": "/lium/corpus/vrac/audio_tagging/", "part": "test", "target_samplerate": args.samplerate}
    elif dataset_name.upper() == "TIMIT":
        cfg_data = {"data_dir": "/lium/corpus/base/TIMIT/","split": "test", "subsplit": "test", "length": 0.5,}
    
    dataset = select_dataset(dataset_name=dataset_name, **cfg_data)
    # evaluate the model
    # this function returns each metric for each file in the dataset
    results = eval(model=model, 
                   dataset=dataset, 
                   device=device, 
                   metrics=["acc", "auc", "f1", "precision", "recall"],)

    print([(i, len(results[i])) for i in results.keys()])
    
    # report the data and save them
    # Convert results dictionary to DataFrame
    out_dir = os.path.join(exp_root,"metrics")
    df = pd.DataFrame(results)

    # Compute mean and std for each metric
    stats = df.agg(['mean', 'std']).to_dict()
    
    # Save per-file metrics
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"metrics_fold{args.dataset_name}.csv"), index=False)

    # Save summary statistics
    with open(os.path.join(out_dir, f"metrics_summary_fold{args.dataset_name}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSummary metrics for ", args.dataset_name)
    print("=" * 40)
    for metric, values in stats.items():
        mean = values.get('mean', None)
        std = values.get('std', None)
        if mean is not None and std is not None:
            print(f"{metric:12s}: mean = {mean:.4f} | std = {std:.4f}")
        else:
            print(f"{metric:12s}: mean = {mean} | std = {std}")
    print("=" * 40)

