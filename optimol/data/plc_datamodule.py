from typing import Any, Dict, Optional, Tuple

import os
import json
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PygDataLoader
from datasets import load_from_disk
import pandas as pd
from optimol.data.components.plc_dataset import PLCDataset
from optimol.data.components.plc_featurize import dock_and_featurize


class PLCDataModule(LightningDataModule):
    def __init__(
        self,
        train_data: str = None,
        test_data: str = None,
        targets_data_dir: str = None,
        ligand_field: str = "ligand",
        complex_id_field: str = "pdb_code",
        target_field: str = "pdb_code",
        target_id: str = None,
        target_path: str = None,
        ligand_prep: Dict[str, Any] = None,
        ligand_dock: Dict[str, Any] = None,
        train_val_split: Tuple[float, float, float] = (0.85, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        keep_in_memory: bool = False,
        pin_memory: bool = True,
        use_protein_global_features: bool = False,
        use_ligand_global_features: bool = False,
        residue_features: str = "none",
        tasks: Dict[str, str] = None,
        equality_only: bool = True,
        output_dir: str = None,
    ) -> None:
        super().__init__()

        if targets_data_dir is None:
            raise ValueError(
                "target_data_dir must be specified. Provide it in the config file."
            )
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[PLCDataset] = None
        self.data_val: Optional[PLCDataset] = None
        self.data_test: Optional[PLCDataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        pass

    def load_dataset(
        self,
        path,
        ligand_field,
        target_field,
        target_path,
        target_id,
        complex_id_field,
        ds_type,
        target_dataset,
        protein_to_index,
        tasks,
        ligand_prep_configs,
        ligand_dock_configs,
        output_dir,
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
            ds, target_dataset, protein_to_index = dock_and_featurize(
                dataframe=ds,
                ligand_field=ligand_field,
                target_field=target_field,
                target_path=target_path,
                target_id=target_id,
                complex_id_field=complex_id_field,
                ds_type=ds_type,
                targets_hfds=target_dataset,
                protein_to_index=protein_to_index,
                tasks=tasks,
                ligand_prep_configs=ligand_prep_configs,
                ligand_dock_configs=ligand_dock_configs,
                output_dir=output_dir,
            )
        return ds, target_dataset, protein_to_index

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
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.targets_data_dir is None:
                raise ValueError(
                    "target_data_dir must be specified. Provide it in the config file."
                )
            else:

                p2i_path = os.path.join(
                    self.hparams.targets_data_dir, "pdb_to_index.json"
                )
                if os.path.exists(p2i_path):
                    with open(p2i_path, "r") as json_file:
                        pdb_to_index = json.load(json_file)
                else:
                    raise ValueError(
                        f"pdb_to_index.json must be present in the target_data_dir at {p2i_path}"
                    )
                targets_ds = load_from_disk(
                    self.hparams.targets_data_dir,
                    keep_in_memory=self.hparams.keep_in_memory,
                )

            if self.hparams.train_data is not None:
                trvl_ds, targets_ds, pdb_to_index = self.load_dataset(
                    self.hparams.train_data,
                    ligand_field=self.hparams.ligand_field,
                    target_field=self.hparams.target_field,
                    target_path=self.hparams.target_path,
                    target_id=self.hparams.target_id,
                    complex_id_field=self.hparams.complex_id_field,
                    ds_type="train",
                    target_dataset=targets_ds,
                    protein_to_index=pdb_to_index,
                    tasks=self.hparams.tasks,
                    ligand_prep_configs=self.hparams.ligand_prep,
                    ligand_dock_configs=self.hparams.ligand_dock,
                    output_dir=self.hparams.output_dir,
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

                    self.data_val = PLCDataset(
                        main_dataset=val_ds,
                        target_dataset=targets_ds,
                        protein_to_index=pdb_to_index,
                        tasks=self.hparams.tasks,
                        target_field=self.hparams.target_field,
                        equality_only=self.hparams.equality_only,
                        use_protein_global_features=self.hparams.use_protein_global_features,
                        use_ligand_global_features=self.hparams.use_ligand_global_features,
                        residue_features=self.hparams.residue_features,
                    )

                self.data_train = PLCDataset(
                    main_dataset=train_ds,
                    target_dataset=targets_ds,
                    protein_to_index=pdb_to_index,
                    tasks=self.hparams.tasks,
                    target_field=self.hparams.target_field,
                    equality_only=self.hparams.equality_only,
                    use_protein_global_features=self.hparams.use_protein_global_features,
                    use_ligand_global_features=self.hparams.use_ligand_global_features,
                    residue_features=self.hparams.residue_features,
                )

            if self.hparams.test_data is not None:
                ts_ds, targets_ds, pdb_to_index = self.load_dataset(
                    self.hparams.test_data,
                    ligand_field=self.hparams.ligand_field,
                    target_field=self.hparams.target_field,
                    target_path=self.hparams.target_path,
                    target_id=self.hparams.target_id,
                    complex_id_field=self.hparams.complex_id_field,
                    ds_type="test",
                    target_dataset=targets_ds,
                    protein_to_index=pdb_to_index,
                    tasks=self.hparams.tasks,
                    ligand_prep_configs=self.hparams.ligand_prep,
                    ligand_dock_configs=self.hparams.ligand_dock,
                    output_dir=self.hparams.output_dir,
                )
                self.data_test = PLCDataset(
                    main_dataset=ts_ds,
                    target_dataset=targets_ds,
                    protein_to_index=pdb_to_index,
                    tasks=self.hparams.tasks,
                    target_field=self.hparams.target_field,
                    equality_only=self.hparams.equality_only,
                    use_protein_global_features=self.hparams.use_protein_global_features,
                    use_ligand_global_features=self.hparams.use_ligand_global_features,
                    residue_features=self.hparams.residue_features,
                )

    def train_dataloader(self):
        return PygDataLoader(
            self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return PygDataLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return PygDataLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

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
