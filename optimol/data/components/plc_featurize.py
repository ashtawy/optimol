import os
import pandas as pd
import logging
import numpy as np
import multiprocessing as mp
import time
from datasets import Dataset, load_from_disk
import json

from data_engine.common.utils import load_molecule, setup_logging
from data_engine.dock.molecule_preparer import MoleculePreparer
from data_engine.dock.docker import Docker
from data_engine.featurize.create_datasets import process_target_light

from sklearn.preprocessing import OneHotEncoder

# TODO: Move most of the functions in this file to the data_engine package

TARGETS_DATA_DIR = "/data/datasets/pdbbind/v2020/v2020-general-PL"
ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "-"]
ATOM_OHE = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
_ = ATOM_OHE.fit([[c] for c in ATOM_TYPES])


def process_batch_sequential(paths_df, overwrite=True):
    out_paths_dfs = []
    molecule_preparer = MoleculePreparer()
    for _, row in paths_df.iterrows():
        input_molecule = row["input_molecule"]
        output_prefix = row["output_prefix"]
        conf_generator = row.get("conf_generator", "balloon")
        n_confs = row.get("n_confs", 1)
        crystal_ligand = row.get("crystal_ligand", None)
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

        out_paths_df = molecule_preparer.embed_and_convert(
            input_molecule=input_molecule,
            output_prefix=output_prefix,
            conformer_generator=conf_generator,
            n_max_conformers=n_confs,
            ref_molecule_path=crystal_ligand,
            ignore_chirality=False,
            n_stereisomers=1,
            overwrite_existing=overwrite,
        )
        if out_paths_df is not None:
            for k, v in row.items():
                if k not in out_paths_df.columns:
                    out_paths_df[k] = v
            out_paths_dfs.append(out_paths_df)

    return out_paths_dfs


def process_batch(
    paths_df,
    conf_generator="balloon",
    n_confs=1,
    crystal_ligand=None,
    overwrite=False,
    num_workers=16,
):
    num_rows = len(paths_df)
    shard_size = num_rows // num_workers
    remainder = num_rows % num_workers

    shards = []
    start = 0
    if "conf_generator" not in paths_df.columns:
        paths_df["conf_generator"] = conf_generator
    if "n_confs" not in paths_df.columns:
        paths_df["n_confs"] = n_confs
    if "crystal_ligand" not in paths_df.columns:
        paths_df["crystal_ligand"] = crystal_ligand
    for i in range(num_workers):
        end = start + shard_size + (1 if i < remainder else 0)
        shards.append(paths_df[start:end])
        start = end

    # Create a multiprocessing pool
    pool = mp.Pool(processes=num_workers)

    # Run each shard in a parallel process
    results = pool.map(process_batch_sequential, shards)

    # Aggregate the returning dictionaries into one dictionary
    status = []
    for result in results:
        if result is not None and len(result) > 0:
            status.extend(result)
    final_result_df = pd.concat(status) if len(status) > 0 else None
    return final_result_df


def prepare_ligands(
    dataframe,
    ligand_field,
    complex_id_field,
    ligand_prep_configs,
    output_dir,
    num_workers,
):
    paths_df = dataframe.copy()
    paths_df["input_molecule"] = paths_df[ligand_field].values
    if output_dir is None or not os.path.exists(output_dir):
        raise ValueError(
            f"Output directory of ligand preparation {output_dir} "
            f"either not provided or does not exist. Please create "
            f"the directory and try again."
        )
    paths_df["output_prefix"] = (
        output_dir + "/" + paths_df[complex_id_field].astype("str")
    )
    conf_generator = ligand_prep_configs["conformer_generators"]
    if not isinstance(conf_generator, str):
        conf_generator = conf_generator[0]
    status_df = process_batch(
        paths_df,
        conf_generator=conf_generator,
        n_confs=ligand_prep_configs["n_confs"],
        crystal_ligand=ligand_prep_configs["crystal_ligand_path"],
        overwrite=ligand_prep_configs["overwrite"],
        num_workers=num_workers,
    )
    conf_summary_path = os.path.join(output_dir, "ligand_prep_summary.csv")
    if status_df is not None:
        status_df.to_csv(conf_summary_path, index=False)
    return status_df


def iterate_in_shards(df, start, stop, shard_size):
    start = max(start, 0)
    stop = min(stop, len(df))
    for i in range(start, stop, shard_size):
        yield df[i : min(i + shard_size, stop)]


def dock(
    docker,
    ligand_list_path,
    log_path,
    num_poses,
    dock_only,
    exhaustiveness,
    num_workers,
):
    # time.sleep(200000)
    summary_df = docker.dock(
        ligand_list_file=ligand_list_path,
        output_dir=os.path.dirname(log_path),
        log_file=log_path,
        mode_count=num_poses,
        extract_energies_only=dock_only,
        search_depth=exhaustiveness,
        num_cpus=num_workers,
    )
    return summary_df


def dock_ligands(
    paths_df,
    ligand_field,
    target_field,
    target_path,
    complex_id_field,
    ligand_dock_configs,
    output_dir,
    num_workers,
):
    paths_df = paths_df.copy()
    if target_field is None and target_path is not None:

        if target_path.endswith(".pdbqt"):
            target_pdbqt = target_path
            target_pdb = target_path.replace(".pdbqt", ".pdb")
        elif target_path.endswith(".pdb"):
            target_pdb = target_path
            target_pdbqt = target_path.replace(".pdb", ".pdbqt")
        else:
            raise ValueError(
                f"Target file {target_path} must be either a pdb or pdbqt file"
            )

        if not os.path.exists(target_pdbqt) and not os.path.exists(target_pdb):
            raise ValueError(f"Both {target_pdbqt} and {target_pdb} do not exist")
        elif not os.path.exists(target_pdbqt):
            raise ValueError(
                f"The PDBQT version of the target file "
                f"{target_pdbqt} does not exist "
                "Please generate pdbqt file using "
                " prepare_receptors.py script in the data_engine "
                "and save it in the same directory as the pdb "
                "file with the same name except for the extension"
            )
        elif not os.path.exists(target_pdb):
            raise ValueError(
                f"The PDB version of the target file {target_pdb} does not exist "
                "Please make sure the pdb file is present in the same directory "
                "as the pdbqt file with the same name except for the extension"
            )
        paths_df["protein"] = target_pdb
        paths_df["protein_pdbqt"] = target_pdbqt
    elif (
        target_field is not None
        and target_field not in paths_df.columns
        and target_path is not None
    ):
        if target_path.endswith(".pdbqt"):
            target_pdbqt = target_path
            target_pdb = target_path.replace(".pdbqt", ".pdb")
        elif target_path.endswith(".pdb"):
            target_pdb = target_path
            target_pdbqt = target_path.replace(".pdb", ".pdbqt")
        else:
            raise ValueError(
                f"Target file {target_path} must be either a pdb or pdbqt file"
            )

        if not os.path.exists(target_pdbqt) and not os.path.exists(target_pdb):
            raise ValueError(f"Both {target_pdbqt} and {target_pdb} do not exist")
        elif not os.path.exists(target_pdbqt):
            raise ValueError(
                f"The PDBQT version of the target file {target_pdbqt} does not exist "
                "Please generate pdbqt file using prepare_receptors.py script in the data_engine "
                "and save it in the same directory as the pdb file with the same name except for the extension"
            )
        elif not os.path.exists(target_pdb):
            raise ValueError(
                f"The PDB version of the target file {target_pdb} does not exist "
                "Please make sure the pdb file is present in the same directory "
                "as the pdbqt file with the same name except for the extension"
            )
        if target_field != "pose_path":
            paths_df[target_field] = target_pdb
        paths_df["protein_pdbqt"] = target_pdbqt
    elif target_field is not None and target_field in paths_df.columns:
        if target_field.endswith("pdbqt") and target_field != "protein_pdbqt":
            paths_df["protein_pdbqt"] = paths_df[target_field]
        elif f"{target_field}_pdbqt" in paths_df.columns:
            paths_df["protein_pdbqt"] = paths_df[f"{target_field}_pdbqt"]
    elif target_field is not None and target_field not in paths_df.columns:
        raise ValueError(
            f"target_field `{target_field}` must be present in the specified file. Present fields in specified file are: {paths_df.columns}"
        )
    else:
        raise ValueError(
            "Either a `target_field` must be populated in the train_data csv file (and target_field must indicate the column name) or a `target_path` must be provided"
        )

    if ligand_field is not None and ligand_field in paths_df.columns:
        paths_df["ligand"] = paths_df[ligand_field]
    elif ligand_field is not None and ligand_field not in paths_df.columns:
        raise ValueError(
            f"ligand_field `{ligand_field}` must be present in the specified file. Present fields in specified file are: {paths_df.columns}"
        )
    elif ligand_field is None:
        raise ValueError(
            "A `ligand_field` must be populated in the csv file (and ligand_field must indicate the column name)"
        )

    paths_df = paths_df.sort_values(by=["protein_pdbqt"])
    if ligand_dock_configs.get("crystal_ligand_path") is not None:
        paths_df["crystal_ligand"] = ligand_dock_configs["crystal_ligand_path"]

    if ligand_dock_configs.get("center_x") is not None:
        paths_df["center_x"] = ligand_dock_configs["center_x"]
    if ligand_dock_configs.get("center_y") is not None:
        paths_df["center_y"] = ligand_dock_configs["center_y"]
    if ligand_dock_configs.get("center_z") is not None:
        paths_df["center_z"] = ligand_dock_configs["center_z"]
    if ligand_dock_configs.get("size_x") is not None:
        paths_df["size_x"] = ligand_dock_configs["size_x"]
    if ligand_dock_configs.get("size_y") is not None:
        paths_df["size_y"] = ligand_dock_configs["size_y"]
    if ligand_dock_configs.get("size_z") is not None:
        paths_df["size_z"] = ligand_dock_configs["size_z"]

    if "center_x" not in paths_df.columns:
        paths_df["center_x"] = None
    if "center_y" not in paths_df.columns:
        paths_df["center_y"] = None
    if "center_z" not in paths_df.columns:
        paths_df["center_z"] = None
    if "size_x" not in paths_df.columns:
        paths_df["size_x"] = None
    if "size_y" not in paths_df.columns:
        paths_df["size_y"] = None
    if "size_z" not in paths_df.columns:
        paths_df["size_z"] = None

    logging_path = os.path.join(output_dir, "docking.log")
    log_path = setup_logging(logging_path)
    overwrite = ligand_dock_configs["overwrite"]

    if "output_path" not in paths_df.columns:
        paths_df["output_path"] = (
            output_dir
            + "/"
            + paths_df["ligand"].str.split("/").str[-1].str.split(".").str[0]
            + "_docked.pdbqt"
        )
    dock_only = ligand_dock_configs.get("dock_only", False)
    docking_engine = ligand_dock_configs["docking_engines"]
    if not isinstance(docking_engine, str):
        docking_engine = docking_engine[0]
    num_poses = ligand_dock_configs["n_poses"]
    exhaustiveness = ligand_dock_configs["exhaustiveness"]
    start_time = time.time()
    summary_dfs = []
    for protein_path in paths_df["protein_pdbqt"].unique():
        df = paths_df[(paths_df["protein_pdbqt"] == protein_path)]

        for j, shard_df in enumerate(iterate_in_shards(df, 0, len(df), 150)):
            crystal_ligand = shard_df["crystal_ligand"].iloc[0]
            center_coords = (
                shard_df[["center_x", "center_y", "center_z"]].iloc[0].tolist()
            )
            box_size = shard_df[["size_x", "size_y", "size_z"]].iloc[0].tolist()

            ligand_list = shard_df[["ligand", "output_path"]].copy()
            ligand_list.rename(
                columns={
                    "ligand": "input_pdbqt_path",
                    "output_path": "output_pdbqt_path",
                },
                inplace=True,
            )
            ligand_list_path = os.path.join(output_dir, f"ligand_list_{j}.csv")

            if not overwrite:
                ligand_list = ligand_list[
                    ~ligand_list["output_pdbqt_path"].apply(os.path.exists)
                ]

            ligand_list.to_csv(ligand_list_path, index=False)
            for lig_output_path in ligand_list["output_pdbqt_path"].unique():
                os.makedirs(os.path.dirname(lig_output_path), exist_ok=True)

            logging.info(
                f"Docking {len(ligand_list)} at {ligand_list_path} ligands to {protein_path}"
            )
            try:
                if docking_engine == "autodock_gpu":
                    protein_path2 = protein_path.replace(".pdbqt", ".maps.fld")
                else:
                    protein_path2 = protein_path
                docker = Docker(
                    receptor_file=protein_path2,
                    cocrystal_file=crystal_ligand,
                    center_coords=center_coords,
                    box_size=box_size,
                    algorithm=docking_engine,
                )
                summary_df = None
                num_tries = 0
                while summary_df is None and num_tries < 3:
                    summary_df = dock(
                        docker,
                        ligand_list_path,
                        log_path,
                        num_poses,
                        dock_only,
                        exhaustiveness,
                        num_workers,
                    )
                    num_tries += 1
                    time.sleep(5)
                if summary_df is not None:
                    summary_df = summary_df.merge(
                        shard_df.rename(columns={"ligand": "input_pdbqt_path"}),
                        on="input_pdbqt_path",
                    )
                    summary_df["protein_pdbqt"] = protein_path
                    if target_field in shard_df.columns:
                        summary_df[target_field] = shard_df[target_field].iloc[0]
                    summary_dfs.append(summary_df)
            except Exception as e:
                logging.error(
                    f"Error occurred while docking ligands at {ligand_list_path} "
                    f"to {protein_path}. Error: {e}"
                )
    end_time = time.time()
    if len(summary_dfs) > 0:
        summary_df = pd.concat(summary_dfs)
        summary_path = os.path.join(output_dir, "docking_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Docking summary table saved to {summary_path}")
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = None
    logging.info(
        f"All done. A total of {len(paths_df)} have been "
        f"docked in {end_time - start_time} seconds. "
    )
    return summary_df


def process_ligand_light(ligand_path):
    molecule, smiles = load_molecule(ligand_path)
    atom_embeds = None
    props = None
    atom_positions = None
    if molecule is not None:
        atom_positions = molecule.GetConformer().GetPositions().astype(np.float16)
        elements = [a.GetSymbol() for a in molecule.GetAtoms()]
        ohe_elements = ATOM_OHE.transform(np.array(elements).reshape([-1, 1]))
        N = ohe_elements.shape[0]
        residue_embeds = np.ones([N, 1])
        # residue_embeds[:, n_residues_incl_lig - 1] = 1.0
        atom_embeds = np.concatenate([ohe_elements, residue_embeds], axis=1).astype(
            bool
        )
        props = 1
        # try:
        #    pftrzr = MolPropertyFeaturizer()
        #    props = pftrzr.featurize(smiles)
        # except Exception as e:
        #    None
    return atom_embeds, atom_positions, smiles, props


def generate_main_ds(df):
    if isinstance(df, list):
        df = pd.concat(df)
    for _, row in df.iterrows():
        ligand_path = row["_ligand_path_to_featurize_"]
        if not os.path.exists(ligand_path):
            ligand_path = ligand_path.replace("smina", "auto")
        # check if the size of the file is larger than 100 bytes
        if os.path.exists(ligand_path) and os.path.getsize(ligand_path) > 100:
            atom_embeds, atom_positions, smiles, props = process_ligand_light(
                ligand_path
            )
        else:
            continue
        passed = (smiles is not None and props is not None) or (
            smiles is None and props is None
        )
        try:
            new_row = {
                "atom_embeddings": atom_embeds,
                "atom_positions": atom_positions,
                "smiles": smiles,
                "props_global_embeddings": props,
            }
            for k, v in row.items():
                new_row[k] = v
            yield new_row
        except Exception as e:
            continue


def to_shards(df, n_shards):
    # The chunk dataframe has 1600000 rows. I would like to create a list of 16 dataframes each with 100000 rows
    dfs = []
    step_size = len(df) // n_shards
    for i in range(0, len(df), step_size):
        st_idx = i
        end_idx = min(i + step_size, len(df))
        dfs.append(df[st_idx:end_idx])
    return dfs


def featurize_ligands(df, output_hfds_directory=None, num_proc=1):
    if num_proc is not None and num_proc <= len(df):
        dfs = to_shards(df, num_proc)
        main_ds = Dataset.from_generator(
            generate_main_ds, gen_kwargs={"df": dfs}, num_proc=num_proc
        )
    else:
        main_ds = Dataset.from_generator(generate_main_ds, gen_kwargs={"df": df})
    if output_hfds_directory is not None:
        os.makedirs(output_hfds_directory, exist_ok=True)
        main_ds.save_to_disk(output_hfds_directory, max_shard_size="10MB")
    return main_ds


def generate_target_ds(df):
    for _, row in df.iterrows():
        target_path = row["_target_path_to_featurize_"]
        ligand_path = row["_ligand_path_to_featurize_"]
        prt_atom_emb, prt_atom_pos, prt_glob_embds = process_target_light(
            target_path, ligand_path
        )
        try:
            new_row = {
                "atom_embeddings": prt_atom_emb,
                "atom_positions": prt_atom_pos,
                "esm_global_embeddings": prt_glob_embds,
            }
            for k, v in row.items():
                new_row[k] = v
            yield new_row
        except:
            raise ValueError(f"Error processing target {target_path}, {ligand_path}")


def featurize_targets(df, output_hfds_directory=None, p2i_json_path=None):
    target_ds = Dataset.from_generator(generate_target_ds, gen_kwargs={"df": df})
    if output_hfds_directory is not None:
        os.makedirs(output_hfds_directory, exist_ok=True)
        target_ds.save_to_disk(output_hfds_directory, max_shard_size="10MB")
    protein_to_index = {r["pose_path"]: i for i, r in enumerate(target_ds)}
    if p2i_json_path is not None:
        with open(p2i_json_path, "w") as json_file:
            json.dump(protein_to_index, json_file)

    return target_ds, protein_to_index


def dock_and_featurize(
    dataframe,
    ligand_field,
    target_field,
    target_path,
    target_id,
    complex_id_field,
    ds_type,
    targets_hfds,
    protein_to_index,
    tasks,
    ligand_prep_configs,
    ligand_dock_configs,
    dock_only=False,
    num_workers=None,
    output_dir=None,
):
    ligands_hfds = None
    protein_to_index = None
    if output_dir is not None:
        ligands_hfds_path = os.path.join(output_dir, "hf_datasets", ds_type, "ligands")
        targets_hfds_path = os.path.join(output_dir, "hf_datasets", ds_type, "targets")
        p2i_path = os.path.join(output_dir, "hf_datasets", "protein_to_index.json")
        if ds_type == "train" and os.path.exists(ligands_hfds_path):
            ligands_hfds = load_from_disk(ligands_hfds_path, keep_in_memory=True)
        if (
            ds_type == "train"
            and os.path.exists(targets_hfds_path)
            and os.path.exists(p2i_path)
        ):
            with open(p2i_path, "r") as json_file:
                protein_to_index = json.load(json_file)

            targets_hfds = load_from_disk(targets_hfds_path, keep_in_memory=True)
    else:
        ligands_hfds_path = None
        targets_hfds_path = None
        p2i_path = None

    if (
        ligands_hfds is not None
        and targets_hfds is not None
        and protein_to_index is not None
    ):
        return ligands_hfds, targets_hfds, protein_to_index

    num_workers = num_workers if num_workers is not None else mp.cpu_count()
    if ligand_field not in dataframe.columns:
        raise ValueError(
            f"ligand_field `{ligand_field}` must be present in the "
            f"specified file. Present fields in specified file are: "
            f"{dataframe.columns}"
        )
    if (
        target_field is not None
        and target_field not in dataframe.columns
        and target_path is None
    ):
        raise ValueError(
            f"target_field `{target_field}` must be present in "
            f"the specified file. Present fields in specified "
            f"file are: {dataframe.columns}"
        )
    if complex_id_field not in dataframe.columns:
        raise ValueError(
            f"complex_id_field `{complex_id_field}` must be present in "
            f"the specified file. Present fields in specified file "
            f"are: {dataframe.columns}"
        )
    if target_id is not None and target_path is None:
        target_path = os.path.join(TARGETS_DATA_DIR, f"{target_id}_protein.pdb")
        crystal_ligand_path = os.path.join(TARGETS_DATA_DIR, f"{target_id}_ligand.sdf")
        if (
            ligand_prep_configs is not None
            and ligand_prep_configs.get("crystal_ligand_path") is None
        ):
            ligand_prep_configs["crystal_ligand_path"] = crystal_ligand_path
        if (
            ligand_dock_configs is not None
            and ligand_dock_configs.get("crystal_ligand_path") is None
        ):
            ligand_dock_configs["crystal_ligand_path"] = crystal_ligand_path

    if ligand_prep_configs is not None and ligand_prep_configs.get("enabled", False):
        output_lig_prep_dir = os.path.join(output_dir, "ligand_prep")
        os.makedirs(output_lig_prep_dir, exist_ok=True)
        dataframe = prepare_ligands(
            dataframe,
            ligand_field,
            complex_id_field,
            ligand_prep_configs,
            output_lig_prep_dir,
            num_workers,
        )
        ligand_field = "output_conformation_path"
        if dataframe is None:
            raise ValueError(
                f"Ligand preparation failed. Please check the log file "
                f"for more information at {output_lig_prep_dir}"
            )
    if ligand_dock_configs is not None and ligand_dock_configs.get("enabled", False):
        output_lig_dock_dir = os.path.join(output_dir, "ligand_dock")
        os.makedirs(output_lig_dock_dir, exist_ok=True)
        dataframe = dock_ligands(
            dataframe,
            ligand_field,
            target_field,
            target_path,
            complex_id_field,
            ligand_dock_configs,
            output_lig_dock_dir,
            num_workers,
        )
        if dataframe is None:
            raise ValueError(
                f"Docking failed. Please check the log file for more information at {output_lig_dock_dir}"
            )
        if ligand_dock_configs.get("dock_only", False):
            dataframe.to_csv(os.path.join(output_dir, "scores.csv.gz"), index=False)
            return None, None, None
        ligand_field = "pose_path"
    dataframe["_ligand_path_to_featurize_"] = dataframe[ligand_field]

    ligands_hfds = featurize_ligands(
        dataframe, output_hfds_directory=ligands_hfds_path, num_proc=num_workers
    )
    if (
        target_id is not None
        and protein_to_index is not None
        and targets_hfds is not None
        and not ligand_dock_configs.get("enabled", False)
        and target_id in protein_to_index
    ):
        return ligands_hfds, targets_hfds, protein_to_index
    else:
        if target_path is not None and os.path.exists(target_path):
            dataframe["_target_path_to_featurize_"] = target_path
        elif target_field is not None and target_field in dataframe.columns:
            dataframe["_target_path_to_featurize_"] = dataframe[target_field]

        if ligand_dock_configs is not None and ligand_dock_configs.get(
            "enabled", False
        ):
            if "pose_path" in dataframe.columns:
                ref_ligand_field = "pose_path"
            else:
                raise ValueError(
                    f"`pose_path` must be present in the dataframe"
                    f"coming out of the docking run. "
                    f"Present fields in specified file are: {dataframe.columns}"
                )
        elif (
            ligand_dock_configs is not None
            and ligand_dock_configs.get("crystal_ligand_path") is not None
        ):
            dataframe["crystal_ligand"] = ligand_dock_configs["crystal_ligand_path"]
            ref_ligand_field = "crystal_ligand"
        elif "crystal_ligand" in dataframe.columns:
            ref_ligand_field = "crystal_ligand"
        elif ligand_field in dataframe.columns and os.path.exists(
            dataframe[ligand_field].iloc[0]
        ):
            ref_ligand_field = ligand_field
        else:
            raise ValueError(
                "A reference ligand path must be provided to featurize the targets"
            )

        dataframe["_reference_ligand_path_"] = dataframe[ref_ligand_field]
        targets_hfds, protein_to_index = featurize_targets(
            dataframe, targets_hfds_path, p2i_path
        )
    return ligands_hfds, targets_hfds, protein_to_index
