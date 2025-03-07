import numpy as np
import torch


def apply_transforms(y, transforms):
    for t in transforms:
        transform_type, params = next(iter(t.items()))
        if transform_type == "log":
            y = np.log10(y)
        elif transform_type == "std":
            y = (y - params["mean"]) / params["std"]
        elif transform_type == "minmax":
            min_val = params["min"]
            max_val = params["max"]
            y = (y - min_val) / (max_val - min_val)
    return y


def extract_yw(row, key, transforms=None):
    missing = row.get(key) is None or np.isnan(row.get(key))
    task_w = key.split("__")[0] + "__w"
    if missing:
        y = 0.0
        w = 0.0
    else:
        y = row[key]
        w = row.get(task_w, 1.0)
        if transforms is not None:
            y = apply_transforms(y, transforms)
    return y, w


def construct_target(main_r, tasks, equality_only, transforms=None):
    n_columns = len(tasks) if equality_only else len(tasks) * 3
    y = torch.zeros([1, n_columns], dtype=torch.float32)
    w = torch.zeros([1, n_columns], dtype=torch.float32)
    for i, task in enumerate(tasks):
        task_transforms = transforms.get(task) if transforms is not None else None
        if equality_only:
            y[0, i], w[0, i] = extract_yw(main_r, task, task_transforms)
        else:
            task_gt = f"{task}__gt"
            task_eq = f"{task}__eq"
            task_eq = task_eq if task_eq in main_r else task
            task_lt = f"{task}__lt"

            y[0, i * 3 + 0], w[0, i * 3 + 0] = extract_yw(main_r, task_gt, task_transforms)
            y[0, i * 3 + 1], w[0, i * 3 + 1] = extract_yw(main_r, task_eq, task_transforms)
            y[0, i * 3 + 2], w[0, i * 3 + 2] = extract_yw(main_r, task_lt, task_transforms)
    return y, w
