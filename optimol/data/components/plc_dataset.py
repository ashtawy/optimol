import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch_geometric.data import Data as PygData
from torch_geometric.data import Dataset as PygDataset

from optimol.data.components.utils import construct_target


class PLCDataset(PygDataset):
    def __init__(
        self,
        main_dataset,
        target_dataset,
        protein_to_index,
        tasks,
        transforms=None,
        target_field=None,
        equality_only=True,
        use_protein_global_features=False,
        use_ligand_global_features=False,
        residue_features=False,
    ):
        super().__init__()
        self.tds = target_dataset
        # self.tds = load_from_disk("/data/datasets/pdbbind/v2020/hf_datasets/targets2")
        self.p2i = protein_to_index
        # breakpoint()
        self.use_protein_global_features = use_protein_global_features
        self.use_ligand_global_features = use_ligand_global_features
        self.residue_features = residue_features
        self.tasks = tasks
        self.transforms = transforms
        self.equality_only = equality_only
        self.target_field = target_field

        self.mds = main_dataset

    def len(self):
        return len(self.mds)

    def get(self, idx):
        main_r = self.mds[idx]
        k = main_r[self.target_field]
        # k = main_r["crystal_ligand"]
        tidx = self.p2i[k]
        target_r = self.tds[tidx]
        # target_r = self.tds[-1]

        atom_features = torch.tensor(
            main_r["atom_embeddings"] + target_r["atom_embeddings"], dtype=torch.float32
        )

        if self.residue_features == "none":
            clm_idxs = list(range(10)) + [36]
            atom_features = atom_features[:, clm_idxs]
        ligand_atom_pos = torch.tensor(main_r["atom_positions"], dtype=torch.float32)

        # Tranlsate ligand atoms by x=100, y=100, z=100
        ligand_atom_pos = ligand_atom_pos + 100.0

        receptor_atom_pos = torch.tensor(
            target_r["atom_positions"], dtype=torch.float32
        )
        # atom_pos = torch.tensor(
        #    main_r["atom_positions"] + target_r["atom_positions"], dtype=torch.float32
        # )
        atom_pos = torch.cat((ligand_atom_pos, receptor_atom_pos), 0)

        graph = PygData(x=atom_features, pos=atom_pos, global_features=None)
        graph.y, graph.w = construct_target(
            main_r, self.tasks, self.equality_only, self.transforms
        )

        exclude_fields = [
            "atom_embeddings",
            "atom_positions",
            "props_global_embeddings",
            "esm_global_embeddings",
        ]
        meta_data = {}
        for k, v in main_r.items():
            if k not in exclude_fields and not isinstance(v, list):
                meta_data[k] = str(v)
        return meta_data, graph
