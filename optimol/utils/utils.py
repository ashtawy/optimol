import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from omegaconf import DictConfig
import numpy as np
from optimol.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def concatenate_dicts(dicts):
    # Initialize an empty dictionary to hold the concatenated results
    concatenated_dict = {}

    # Iterate over each dictionary in the list
    for d in dicts:
        for key, value in d.items():
            if key in concatenated_dict:
                # If the key is already in the result dictionary, concatenate the lists
                concatenated_dict[key].extend(value)
            else:
                # If the key is not in the result dictionary, initialize it with the current list
                concatenated_dict[key] = value.copy()  # Use copy to avoid reference issues
    return concatenated_dict

def collate(batches):
    meta_data = []
    preds = []
    labels = []
    weights = []
    for batch in batches:
        meta_data.append(batch[0])
        preds.append(batch[1])
        labels.append(batch[2])
        weights.append(batch[3])
    meta_data = concatenate_dicts(meta_data)
    labels = None if labels[0] is None else torch.cat(labels)
    weights = None if weights[0] is None else torch.cat(weights)
    return meta_data, torch.cat(preds), labels, weights


def apply_inverse_transforms(y, transforms):
    for t in transforms[::-1]:
        transform_type, params = next(iter(t.items()))
        if transform_type == "log":
            y = np.power(10, y)
        elif transform_type == "std":
            y = (y * params["std"]) + params["mean"]
        elif transform_type == "minmax":
            min_val = params["min"]
            max_val = params["max"]
            y = y * (max_val - min_val) + min_val
    return y

def inverse_transform(df, tasks, transforms):
    for t, (task_name, task_mode) in enumerate(tasks.items()):
        task_transforms = transforms.get(task_name)
        if task_transforms is not None:
            t_clms = [f"{task_name}_predicted",
                          f"{task_name}_observed",
                          f"{task_name}_observed__eq",
                          f"{task_name}_observed__gt",
                          f"{task_name}_observed__lt"]
            for t_clm in t_clms:
                if t_clm in df.columns:
                    non_missing = ~df[t_clm].isna()
                    if sum(non_missing) > 0:
                        df.loc[non_missing, t_clm] = apply_inverse_transforms(df.loc[non_missing, t_clm].values,
                                                                                task_transforms)
    return df