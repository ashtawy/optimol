from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.regression import MeanSquaredError, SpearmanCorrCoef

from optimol.models.components.mtask_loss import MultiTaskLoss
from optimol.models.components.mtask_metrics import MultiTaskMetrics


class GnnLitModule(LightningModule):
    """Example of a `LightningModule` for GNN predcition.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        ensemble_size: int,
        tasks: Dict[str, str],
    ) -> None:
        """Initialize a `GnnLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        task_loss_types = [
            "bce" if task == "classification" else "mse" for task in tasks.values()
        ]
        task_metric_types = [
            "auc" if task == "classification" else "spearman" for task in tasks.values()
        ]
        self.criterion = MultiTaskLoss(task_loss_types=task_loss_types, device="cuda")
        self.metrics = MultiTaskMetrics(
            task_metric_types=task_metric_types, device="cuda"
        )
        # metric objects for calculating and averaging accuracy across batches
        self.train_perf = MeanMetric()
        self.val_perf = MeanMetric()
        self.test_perf = MeanMetric()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_perf_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_perf.reset()
        self.val_perf_best.reset()
        self.train_loss.reset()
        self.train_perf.reset()
        self.metrics.reset()  # Reset accumulated metrics

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        meta_data, xyw = batch
        preds = self.forward(xyw)
        total_loss, task_losses = self.criterion(preds, xyw.y, xyw.w)
        return total_loss, task_losses, preds, xyw.y, xyw.w

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        total_loss, task_losses, preds, targets, weights = self.model_step(batch)
        # total_perf, task_perf = self.metrics(preds, targets, weights)
        total_perf = 1.0
        # update and log metrics
        self.train_loss(total_loss)
        self.train_perf(total_perf)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/perf", self.train_perf, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return total_loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.train_perf.reset()
        self.train_loss.reset()
        self.metrics.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        total_loss, task_losses, preds, targets, weights = self.model_step(batch)
        total_perf, task_perf = self.metrics(preds, targets, weights)
        # update and log metrics
        self.val_loss(total_loss)
        self.val_perf(total_perf)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/perf", self.val_perf, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        perf = self.val_perf.compute()  # get current val acc
        self.val_perf_best(perf)  # update best so far val acc
        # log `val_corr_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/perf_best", self.val_perf_best.compute(), sync_dist=True, prog_bar=True
        )

        # Reset metrics
        self.val_perf.reset()
        self.val_loss.reset()
        self.metrics.reset()  # Reset accumulated metrics

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        total_loss, task_losses, preds, targets, weights = self.model_step(batch)
        total_perf, task_perf = self.metrics(preds, targets, weights)

        # update and log metrics
        self.test_loss(total_loss)
        self.test_perf(total_perf)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/perf", self.test_perf, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        meta_data, xyw = batch
        preds = self.forward(xyw)
        return meta_data, preds, xyw.y, xyw.w

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = GnnLitModule(None, None, None, None)
