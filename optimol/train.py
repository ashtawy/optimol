from typing import Any, Dict, List, Optional, Tuple
import os
import hydra
import lightning as L
import rootutils
import torch
import numpy as np
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import warnings
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
from optimol.utils.metrics import generate_perf_report
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from optimol.utils import (
    RankedLogger,
    collate,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def update_config(cfg):
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.tasks = cfg.data.tasks
    cfg.model.net.n_tasks = len(cfg.data.tasks)

    OmegaConf.set_struct(cfg.model, True)

    OmegaConf.set_struct(cfg.data, False)
    if cfg.data.get("output_dir") is None:
        cfg.data.output_dir = cfg.paths.output_dir
    OmegaConf.set_struct(cfg.data, False)

    return cfg

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    ensemble_models = []
    ensemble_trainers = []
    tr_default_root_dir = cfg.trainer.default_root_dir
    call_backs_model_cpt_path = cfg.callbacks.model_checkpoint.dirpath
    cfg = update_config(cfg)

    for i in range(cfg.model.get('ensemble_size', 1)):
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

        log.info("Instantiating callbacks...")
        cfg.callbacks.model_checkpoint.dirpath = os.path.join(call_backs_model_cpt_path, f"model_{i+1}", "checkpoints")
        os.makedirs(cfg.callbacks.model_checkpoint.dirpath, exist_ok=True)
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

        log.info(f"Instantiating model {i+1} <{cfg.data._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info(f"Instantiating trainer {i+1} <{cfg.trainer._target_}>")
        cfg.trainer.default_root_dir = os.path.join(tr_default_root_dir, f"model_{i+1}")
        os.makedirs(cfg.trainer.default_root_dir, exist_ok=True)
        #breakpoint()
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Logging hyperparameters!")
            log_hyperparameters(object_dict)

        if cfg.get("train"):
            log.info(f"Starting training model {i+1}!")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        ensemble_models.append(model)
        ensemble_trainers.append(trainer)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing ensemble!")
        test_results = []
        all_predictions = None
        for model, trainer in zip(ensemble_models, ensemble_trainers):
            # trainer.test(model=model, datamodule=datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path)
            batches = trainer.predict(model=model, dataloaders=datamodule.test_dataloader())
            meta_data, predictions, y, weights = collate(batches)
            all_predictions = predictions if all_predictions is None else all_predictions + predictions
            test_results.append(trainer.callback_metrics)
            

        avg_predictions = all_predictions / len(ensemble_models)
        pred_columns = [f"{t}_predicted" for t in cfg.data.tasks.keys()]
        obs_columns = [f"{t}_observed" for t in cfg.data.tasks.keys()]
        df = pd.DataFrame(data=avg_predictions.cpu().numpy(), columns=pred_columns)
        # cover logits to probabilities for classification tasks
        for task_name, task_mode in cfg.data.tasks.items():
            if task_mode == "classification":
                df[f"{task_name}_predicted"] = 1./(1. + np.exp(-df[f"{task_name}_predicted"].values))
        if y is not None:
            y = y.cpu().numpy()
            weights = weights.cpu().numpy()
            y[weights < 1e-9] = np.nan
            if y.shape[1] == len(obs_columns):
                tdf = pd.DataFrame(data=y, columns=obs_columns)
            elif y.shape[1] == 3*len(cfg.data.tasks.keys()):
                new_obs_columns = [[f"{t}__gt", f"{t}__eq", f"{t}__lt"] for t in obs_columns]
                new_obs_columns = [item for sublist in new_obs_columns for item in sublist]
                tdf = pd.DataFrame(data=y.cpu().numpy(), columns=new_obs_columns)
            else:
                raise ValueError("The number of observed columns does not match the number of tasks")
            df = pd.concat([df, tdf], axis=1)

        
        df = pd.concat([df, pd.DataFrame(meta_data)], axis=1)
        df.to_csv(os.path.join(tr_default_root_dir, "test_scores.csv"), index=False)
        if y is not None:
            test_metrics = {}
            test_perf = generate_perf_report(df, weights, cfg.data.tasks, "test")
            
            test_metrics["mse"] = test_perf["mse"].mean()
            test_metrics["pearsonr"] = test_perf["pearsonr"].mean()
            test_metrics["spearmanr"] = test_perf["spearmanr"].mean()
            test_metrics["auc"] = test_perf["auc"].mean()
            test_metrics["acc"] = test_perf["acc"].mean()
            
            log.info("Averaged test metrics: {}".format(test_metrics))
        
            test_perf.to_csv(os.path.join(tr_default_root_dir, "test_perf.csv"), index=False)
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
