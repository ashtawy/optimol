import torch
import torch.nn.functional as F
from sklearn.metrics import (f1_score, accuracy_score, 
                             matthews_corrcoef,
                             mean_absolute_error, mean_squared_error,
                             roc_auc_score)
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd

class Accuracy:
    def __init__(self):
        pass
    def __call__(self, predictions, dataloader):
        all_preds = []
        all_labels = []
        for meta_data, xy in dataloader:
            inputs, labels = xy
            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return accuracy_score(all_labels, all_preds)

class F1Score:
    def __init__(self):
        pass
    def __call__(self, predictions, dataloader, average='macro'):
        all_preds = []
        all_labels = []
        for meta_data, xy in dataloader:
            inputs, labels = xy
            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return f1_score(all_labels, all_preds, average=average)

class MSE:
    def __init__(self):
        pass
    
    def __call__(self, predictions, dataloader):
        all_preds = predictions.cpu().numpy()
        all_labels = []
        #all_smiles = []
        #all_name = []
        #all_inchi_key = []
        for meta_data, xy in dataloader:
            all_labels.extend(xy.y.cpu().numpy())
            #all_smiles.extend(batch.smiles.cpu().numpy())
            #all_name.extend(batch.name.cpu().numpy())
            #all_inchi_key.extend(batch.inchi_key.cpu().numpy())
        all_labels = np.array(all_labels)
        #all_smiles = np.array(all_smiles)
        #all_name = np.array(all_name)
        #all_inchi_key = np.array(all_inchi_key)
        df = pd.DataFrame({"preds": all_preds, "labels": all_labels})
        return df, F.mse_loss(torch.tensor(all_preds), torch.tensor(all_labels)).item()


class Pearsonr:
    def __init__(self):
        pass
    
    def __call__(self, predictions, dataloader):
        all_preds = predictions.cpu().numpy()
        all_labels = []
        meta_data_l = []
        for meta_data, xy in dataloader:
            all_labels.extend(xy.y.cpu().numpy())
            meta_data_l.append(meta_data)
        all_labels = np.array(all_labels)
        df = pd.DataFrame({"preds": all_preds, "labels": all_labels})
        return df, pearsonr(all_preds, all_labels)[0]


class Spearmanr:
    def __init__(self):
        pass
    
    def __call__(self, predictions, dataloader):
        all_preds = predictions.cpu().numpy()
        all_labels = []
        #all_smiles = []
        #all_name = []
        #all_inchi_key = []
        for _, xy in dataloader:
            all_labels.extend(xy.y.cpu().numpy())
            #all_smiles.extend(batch.smiles.cpu().numpy())
            #all_name.extend(batch.name.cpu().numpy())
            #all_inchi_key.extend(batch.inchi_key.cpu().numpy())
        all_labels = np.array(all_labels)
        #all_smiles = np.array(all_smiles)
        #all_name = np.array(all_name)
        #all_inchi_key = np.array(all_inchi_key)
        df = pd.DataFrame({"preds": all_preds, "labels": all_labels})
        return df, spearmanr(all_preds, all_labels)[0]

def calculate_enrichment(y, yhat, top_p=0.1):
    scores = pd.DataFrame({"y": y, "yhat": yhat})
    scores = scores.sort_values(by="yhat", ascending=False)
    Ns = int(scores.shape[0] * top_p)
    ns = scores.iloc[:Ns]["y"].sum()
    N = scores.shape[0]
    n = scores["y"].sum()
    ef = (ns / Ns) / (n / N)
    return ef

def calculate_pairwise_success_rate(ground_truth, predictions, max_n=10000):
    if len(ground_truth) != len(predictions):
        raise ValueError("The length of predictions must match the number of rows in the ground truth matrix.")

    total_pairs = 0
    successful_pairs = 0
    if len(ground_truth) > max_n:
        return None
    # Iterate through all possible pairs
    for i in range(len(predictions)):
        for j in range(len(predictions)):
            if i != j:
                # Determine the relationship in predictions
                pred_relation = np.sign(predictions[i] - predictions[j])

                # Determine the relationship in ground truth
                gt_relation = np.sign(ground_truth[i] - ground_truth[j])

                # Check if the relationship matches
                if pred_relation == gt_relation:
                    successful_pairs += 1

                total_pairs += 1

    # Calculate success rate
    success_rate = successful_pairs / total_pairs if total_pairs > 0 else 0
    return success_rate

def get_performance(y, yhat, ds_type, task_name, task_mode):
    perf = {}
    perf["ds_type"] = ds_type
    perf["task"] = task_name
    perf["n_eq_examples"] = len(y)
    perf["y_avg"] = np.mean(y)
    perf["mse"] = None
    perf["mae"] = None
    perf["pearsonr"] = None
    perf["spearmanr"] = None
    perf["pw_rnk_scs_rate"] = None
    perf["auc"] = None
    perf["acc"] = None
    perf["mcc"] = None
    perf["ef_1p"] = None
    if y is not None:
        if task_mode == "regression":
            mse = mean_squared_error(yhat, y)
            mae = mean_absolute_error(yhat, y)
            cor_r = pearsonr(yhat, y.flatten())[0]
            cor_s = spearmanr(yhat, y.flatten())[0]
            perf["mse"] = mse
            perf["mae"] = mae
            perf["pearsonr"] = cor_r
            perf["spearmanr"] = cor_s
            perf["pw_rnk_scs_rate"] = calculate_pairwise_success_rate(y.flatten().round(2), yhat.flatten().round(2))
        else:
            try:
                auc = roc_auc_score(y, yhat)
            except Exception as E:
                auc = np.nan
                print(E)
            yhat_binary = np.zeros_like(yhat)
            yhat_binary[yhat > 0.5] = 1
            try:
                acc = accuracy_score(y, yhat_binary)
            except Exception as E:
                acc = np.nan
            try:
                mcc = matthews_corrcoef(y, yhat_binary)
            except Exception as E:
                mcc = np.nan
            try:
                ef_1p = calculate_enrichment(y, yhat, top_p=0.01)
            except Exception as E:
                ef_1p = np.nan
            perf["auc"] = auc
            perf["acc"] = acc
            perf["mcc"] = mcc
            perf["ef_1p"] = ef_1p

    perf_df = pd.DataFrame([perf])
    perf_df = perf_df.round(4)
    return perf_df

def generate_perf_report(scores, weights, tasks, ds_type):
    perfs = []
    is_inequality = weights.shape[1] == 3 * len(tasks)
    for t, (task_name, task_mode) in enumerate(tasks.items()):
        t_observed = f"{task_name}_observed__eq" if is_inequality else f"{task_name}_observed"
        t_predicted = f"{task_name}_predicted"
        t_index = t*3 + 1 if is_inequality else t
        t_w = weights[:, t_index]
        v_idxs = t_w > 0
        if sum(v_idxs) > 1:
            y = scores[t_observed].loc[v_idxs].values
            yhat = scores[t_predicted].loc[v_idxs].values
            perf = get_performance(y, yhat, ds_type, task_name, task_mode)
            perfs.append(perf)

    perf_df = pd.concat(perfs) if perfs else None
    return perf_df
