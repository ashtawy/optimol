import numpy as np
from torch_geometric.data import Data as PygData, Dataset as PygDataset
import torch
from  optimol.data.components.utils import construct_target
import os
import logging


class Mol2dDataset(PygDataset):
    def __init__(
        self,
        main_dataset,
        tasks,
        transforms,
        equality_only=True
    ):
        super().__init__()
        self.tasks = tasks
        self.equality_only = equality_only
        self.transforms = transforms
        self.hfds = main_dataset

    def len(self):
        return len(self.hfds)

    def get(self, idx):
        row = self.hfds[idx]
        atom_features = np.array(row["node_features"], dtype=np.float32)
        edge_index = np.array(row["edge_index"], dtype=np.int64)
        edge_features = np.array(row["edge_features"], dtype=np.float32)

        global_features = np.array(row["global_features"], dtype=np.float32).reshape(
            [1, -1]
        )
        graph = PygData(
            x=torch.from_numpy(atom_features).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_features.astype(np.float32)),
            global_features=torch.from_numpy(global_features),
        )
        graph.y, graph.w = construct_target(row, self.tasks, self.equality_only, self.transforms)

        exclude_fields = [
            "atom_embeddings",
            "atom_positions",
            "props_global_embeddings",
            "esm_global_embeddings",
        ]
        meta_data = {}
        for k, v in row.items():
            if k not in exclude_fields and not isinstance(v, list):
                meta_data[k] = str(v)
        return meta_data, graph
