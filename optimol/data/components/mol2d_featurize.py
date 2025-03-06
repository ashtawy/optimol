import numpy as np
from torch_geometric.data import Data as PygData, Dataset as PygDataset
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from data_engine.featurize.mols_2d import MolPropertyFeaturizer, Mol2DGraphFeaturizer
from optimol.data.components.utils import construct_target
import os
import logging


def generate_features_light(df, smiles_column="smiles"):
    if isinstance(df, list):
        df = pd.concat(df)
    pftrzr = MolPropertyFeaturizer()
    gftrzr = Mol2DGraphFeaturizer()
    for _, row in df.iterrows():
        smiles = row[smiles_column]
        smiles = max(smiles.split("."), key=len)
        if smiles is not None:
            try:
                pftrs = pftrzr.featurize(smiles)
                gftrs = gftrzr.featurize(smiles)
                gftrs["global_features"] = pftrs
                gftrs["smiles"] = smiles
                for k, v in row.items():
                    if k != smiles_column:
                        gftrs[k] = v
                yield gftrs
            except Exception as e:
                pass


def to_shards(df, n_shards):
    # The chunk dataframe has 1600000 rows. I would like to create a list of 16 dataframes each with 100000 rows
    dfs = []
    step_size = len(df) // n_shards
    for i in range(0, len(df), step_size):
        st_idx = i
        end_idx = min(i + step_size, len(df))
        dfs.append(df[st_idx:end_idx])
    return dfs


def featurize(
    dataframe, ligand_field, ligand_id_field, ds_type, output_dir=None, num_proc=1
):
    clms = dataframe.columns.tolist()
    if ligand_field is None:
        raise ValueError(
            f"ligand_field is required (e.g., `smiles`). " f"Existing columns: {clms}"
        )
    elif ligand_field not in dataframe.columns:
        raise ValueError(
            f"ligand_field {ligand_field} not in dataframe columns. "
            f"Existing columns: {clms}"
        )

    if ligand_id_field is None:
        raise ValueError(
            f"ligand_id_field is required (e.g., `inchi_key` or `name`). "
            f"Existing columns: {clms}"
        )
    elif ligand_id_field not in dataframe.columns:
        raise ValueError(
            f"ligand_id_field {ligand_id_field} not in dataframe columns. "
            f"Existing columns: {clms}"
        )
    hfds_path = None
    hfds = None
    if output_dir is not None:
        hfds_path = os.path.join(output_dir, "hf_datasets", ds_type, "ligands")
        # We ignore the test dataset to force refeaturization in case the user made a mistake
        # in the first inference run by providing the wrong test set and corrected themselves
        # but executed in the same output directory
        if ds_type.lower() != "test" and os.path.exists(hfds_path):
            try:
                hfds = load_from_disk(hfds_path, keep_in_memory=True)
            except Exception as e:
                logging.error(
                    f"Error loading huggingface dataset from {hfds_path}: {e}"
                )
                hfds = None

    if hfds is not None:
        return hfds

    if num_proc is not None and num_proc <= len(dataframe):
        dataframes = to_shards(dataframe, num_proc)
        hfds = Dataset.from_generator(
            generate_features_light,
            gen_kwargs={"df": dataframes, "smiles_column": ligand_field},
            num_proc=num_proc,
        )
    else:
        hfds = Dataset.from_generator(
            generate_features_light,
            gen_kwargs={"df": dataframe, "smiles_column": ligand_field},
        )
    if hfds_path is not None:
        os.makedirs(hfds_path, exist_ok=True)
        hfds.save_to_disk(hfds_path)
    return hfds
