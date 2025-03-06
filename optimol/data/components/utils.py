import numpy as np
import torch


def extract_yw(row, key):
    missing = row.get(key) is None or np.isnan(row.get(key))
    task_w = key.split("__")[0] + "__w"
    y = 0.0 if missing else row[key]
    w = 0.0 if missing else row.get(task_w, 1.0)
    return y, w


def construct_target(main_r, tasks, equality_only):
    n_columns = len(tasks) if equality_only else len(tasks) * 3
    y = torch.zeros([1, n_columns], dtype=torch.float32)
    w = torch.zeros([1, n_columns], dtype=torch.float32)
    for i, task in enumerate(tasks):
        if equality_only:
            y[0, i], w[0, i] = extract_yw(main_r, task)
        else:
            task_gt = f"{task}__gt"
            task_eq = f"{task}__eq"
            task_eq = task_eq if task_eq in main_r else task
            task_lt = f"{task}__lt"

            y[0, i * 3 + 0], w[0, i * 3 + 0] = extract(main_r, task_gt)
            y[0, i * 3 + 1], w[0, i * 3 + 1] = extract(main_r, task_eq)
            y[0, i * 3 + 2], w[0, i * 3 + 2] = extract(main_r, task_lt)
    return y, w
