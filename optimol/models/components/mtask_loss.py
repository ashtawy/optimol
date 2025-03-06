from collections import defaultdict
import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, task_loss_types, device, verbose=False):
        super(MultiTaskLoss, self).__init__()
        self.task_loss_types = task_loss_types if isinstance(task_loss_types, list) else [task_loss_types]
        self.num_tasks = len(self.task_loss_types)
        self.verbose = verbose
        self.device = device

        self.loss_functions = {
            "bce": nn.BCEWithLogitsLoss(reduction="none"),
            "mse": nn.MSELoss(reduction="none"),
        }

        self.compact_indices, self.expanded_indices = self._create_task_indices()

    def _create_task_indices(self):
        compact_indices = defaultdict(list)
        expanded_indices = defaultdict(list)

        for task_idx, loss_type in enumerate(self.task_loss_types):
            compact_indices[loss_type].append(task_idx)
            expanded_indices[loss_type].extend(range(task_idx * 3, task_idx * 3 + 3))

        for loss_type in self.task_loss_types:
            compact_indices[loss_type] = torch.tensor(compact_indices[loss_type], device=self.device)
            expanded_indices[loss_type] = torch.tensor(expanded_indices[loss_type], device=self.device)

        return compact_indices, expanded_indices

    @staticmethod
    def compress_tensor(tensor, reduction="sum"):
        batch_size, num_columns = tensor.shape
        if num_columns % 3 != 0:
            raise ValueError("The number of columns must be a multiple of 3.")
        
        reshaped = tensor.view(batch_size, num_columns // 3, 3)
        if reduction == "sum":
            return reshaped.sum(dim=2)
        elif reduction == "max":
            return torch.amax(reshaped, dim=2)
        else:
            raise ValueError(f"Unsupported reduction operation: {reduction}")

    def _apply_censored_mse(self, predictions, targets, loss_tensor):
        greater_than_mask = torch.ones_like(targets)
        less_than_mask = torch.ones_like(targets)
        
        greater_than_indices = torch.arange(0, targets.shape[1], 3, device=targets.device)
        less_than_indices = torch.arange(2, targets.shape[1], 3, device=targets.device)
        
        greater_than_mask[:, greater_than_indices] = (predictions[:, greater_than_indices] < targets[:, greater_than_indices]).float()
        less_than_mask[:, less_than_indices] = (predictions[:, less_than_indices] > targets[:, less_than_indices]).float()
        
        return loss_tensor * greater_than_mask * less_than_mask

    def forward(self, predictions, targets, weights):
        is_compact_format = targets.shape[1] == self.num_tasks
        task_indices = self.compact_indices if is_compact_format else self.expanded_indices
        predictions = predictions if is_compact_format else predictions.repeat_interleave(3, dim=1)

        task_losses = torch.zeros(self.num_tasks, device=targets.device, dtype=targets.dtype)

        for loss_type, indices in task_indices.items():
            compact_indices = self.compact_indices[loss_type]
            task_predictions = predictions[:, indices]
            task_targets = targets[:, indices]
            task_weights = weights[:, indices]
            
            loss_values = self.loss_functions[loss_type](task_predictions, task_targets) * task_weights

            if loss_type == "mse" and not is_compact_format:
                loss_values = self._apply_censored_mse(task_predictions, task_targets, loss_values)

            loss_values = loss_values if is_compact_format else self.compress_tensor(loss_values)
            task_losses.scatter_(0, compact_indices, loss_values.sum(dim=0))

        weights = weights if is_compact_format else self.compress_tensor(weights, "max")
        task_losses /= weights.sum()
        total_loss = task_losses.sum()

        return total_loss, task_losses