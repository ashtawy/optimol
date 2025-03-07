from typing import Any, Dict, Optional, Tuple

import os
import json
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PygDataLoader
from datasets import load_from_disk
import datetime
import pandas as pd
from optimol.data.components.mol2d_dataset import Mol2dDataset
from optimol.data.components.mol2d_featurize import featurize


class Mol2dDataModule(LightningDataModule):
    def __init__(
        self,
        train_data: str = None,
        test_data: str = None,
        train_val_split: Tuple[float, float, float] = (0.85, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        keep_in_memory: bool = False,
        pin_memory: bool = False,
        ligand_field: str = None,
        ligand_id_field: str = None,
        tasks: Dict[str, str] = None,
        transforms: Dict[str, Dict[str, Any]] = None,
        equality_only: bool = True,
        output_dir: str = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Mol2dDataset] = None
        self.data_val: Optional[Mol2dDataset] = None
        self.data_test: Optional[Mol2dDataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        pass

    def load_dataset(
        self, path, ligand_field, ligand_id_field, ds_type, output_dir, num_workers
    ):
        ds = None
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")

        if os.path.isdir(path):
            try:
                ds = load_from_disk(path, keep_in_memory=False)
            except Exception as e:
                raise ValueError(f"Error loading huggingface dataset from {path}: {e}")
        elif os.path.isfile(path):
            try:
                ds = pd.read_csv(path)
            except Exception as e:
                raise ValueError(f"Error loading csv dataset from {path}: {e}")

        if ds is not None and isinstance(ds, pd.DataFrame):
            ds = featurize(
                dataframe=ds,
                ligand_field=ligand_field,
                ligand_id_field=ligand_id_field,
                ds_type=ds_type,
                output_dir=output_dir,
                num_proc=num_workers,
            )
        return ds

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already

        if not self.data_train and self.hparams.train_data is not None:
            trvl_ds = self.load_dataset(
                self.hparams.train_data,
                ligand_field=self.hparams.ligand_field,
                ligand_id_field=self.hparams.ligand_id_field,
                ds_type="train",
                output_dir=self.hparams.output_dir,
                num_workers=self.hparams.num_workers,
            )
            train_ds = trvl_ds
            self.data_val = None
            if (
                self.hparams.train_val_split is not None
                and self.hparams.train_val_split[0] < 1.0
            ):
                trvl_ds = trvl_ds.train_test_split(
                    train_size=self.hparams.train_val_split[0]
                )
                train_ds = trvl_ds["train"]
                val_ds = trvl_ds["test"]

                self.data_val = Mol2dDataset(
                    main_dataset=val_ds,
                    tasks=self.hparams.tasks,
                    transforms=self.hparams.transforms,
                    equality_only=self.hparams.equality_only,
                )

            self.data_train = Mol2dDataset(
                main_dataset=train_ds,
                tasks=self.hparams.tasks,
                transforms=self.hparams.transforms,
                equality_only=self.hparams.equality_only,
            )

        if not self.data_test and self.hparams.test_data is not None:
            ts_ds = self.load_dataset(
                self.hparams.test_data,
                ligand_field=self.hparams.ligand_field,
                ligand_id_field=self.hparams.ligand_id_field,
                ds_type="test",
                output_dir=self.hparams.output_dir,
                num_workers=self.hparams.num_workers,
            )
            self.data_test = Mol2dDataset(
                main_dataset=ts_ds,
                tasks=self.hparams.tasks,
                transforms=self.hparams.transforms,
                equality_only=self.hparams.equality_only,
            )
        elif self.hparams.test_data is None:
            self.data_test = None

    def train_dataloader(self):
        return PygDataLoader(
            self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.data_val is None:
            return None
        return PygDataLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        dl = PygDataLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        return dl

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
