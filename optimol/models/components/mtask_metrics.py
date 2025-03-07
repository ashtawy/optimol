import torch
import torch.nn as nn
from collections import defaultdict
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef

class MultiTaskMetrics(nn.Module):
    def __init__(self, task_metric_types, device):
        super(MultiTaskMetrics, self).__init__()
        self.task_metric_types = task_metric_types if isinstance(task_metric_types, list) else [task_metric_types]
        self.num_tasks = len(self.task_metric_types)
        self.device = device

        self.auroc = BinaryAUROC().to(device)
        self.spearman = SpearmanCorrCoef().to(device)
        self.metric_functions = {
            "spearman": self._compute_regression_metrics,
            "auc": self._compute_auc,
        }
    def reset(self):
        self.auroc.reset()
        self.spearman.reset()
    def _compute_auc(self, predictions, targets, weights):
        if self.is_compact_format:
            mask = weights[:, 0] > 0
            filtered_targets = targets[mask, 0]
        else:
            mask = weights[:, 1] > 0
            filtered_targets = targets[mask, 1]
        filtered_preds = predictions[mask]
        
        if filtered_targets.unique().numel() < 2:
            return torch.tensor(0.5, device=self.device)
        
        return self.auroc(filtered_preds, filtered_targets.long())

    def _compute_regression_metrics(self, predictions, targets, weights):
        # Spearman correlation for equality (=) cases
        if self.is_compact_format:
            eq_mask = weights[:, 0] > 0
            eq_targets = targets[eq_mask, 0]
        else:
            eq_mask = weights[:, 1] > 0
            eq_targets = targets[eq_mask, 1]
        eq_preds = predictions[eq_mask]
        if eq_mask.sum() > 1:
            correlation = self.spearman(eq_preds, eq_targets)
        else:
            correlation = torch.tensor(0.0, device=self.device)

        # Percentage of correct predictions for inequality cases
        #greater_than_mask = weights[:, 0] > 0
        #less_than_mask = weights[:, 2] > 0
        
        #less_than_correct = (predictions[less_than_mask, 0] < targets[less_than_mask, 0]).float().mean() if less_than_mask.any() else torch.tensor(1.0, device=self.device)
        #greater_than_correct = (predictions[greater_than_mask, 2] > targets[greater_than_mask, 2]).float().mean() if greater_than_mask.any() else torch.tensor(1.0, device=self.device)
        
        #inequality_percentage = (less_than_correct + greater_than_correct) / 2

        return correlation # (correlation + inequality_percentage) / 2

    def forward(self, predictions, targets, weights):
        predictions = predictions[0] if isinstance(predictions, list) else predictions
        targets = targets[0] if isinstance(targets, list) else targets
        weights = weights[0] if isinstance(weights, list) else weights

        self.is_compact_format = targets.shape[1] == self.num_tasks
        

        task_metrics = torch.zeros(self.num_tasks, device=self.device)

        for index, metric_type in enumerate(self.task_metric_types):
            task_predictions = predictions[:, index]
            idxs = [index] if self.is_compact_format else list(range(index*3, index*3+3))
            task_targets = targets[:, idxs]
            task_weights = weights[:, idxs]
            
            metric_values = self.metric_functions[metric_type](task_predictions, task_targets, task_weights)
            task_metrics[index] = metric_values

        average_metric = task_metrics.mean()

        return average_metric, task_metrics