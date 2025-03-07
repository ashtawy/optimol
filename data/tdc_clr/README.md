# TDC Microsomal Clearance Benchmark

This is a simple benchmark experiment using experimental microsomal clearance data from the Therapeutics Data Commons (TDC) to quickly test a ligand-based model.

The experiment follows a similar approach to the one described in [this blog post](https://www.inductive.bio/blog/building-better-benchmarks-for-adme-optimization) by Inductive Bio, where they trained and tested a ChemProp (GNN) model, SVR, and a logP baseline. We used the same train/test splits as described in their tutorial for direct comparison.

## Running the Experiment

To train and test the model, run:

```bash
# assuming you copied the files tdc_microsome_train_val.csv & tdc_microsome_test.csv in this repo to /tmp/tdc_clearance
optimol_train data=mol2d 
            data.train_data=/tmp/tdc_clearance/tdc_microsome_train_val.csv \
            data.test_data=/tmp/tdc_clearance/tdc_microsome_test.csv \
            data.ligand_field=smiles \
            data.ligand_id_field=name \
            data.tasks='{Y_log:regression}' \
            model.ensemble_size=5 \
            hydra.run.dir=/tmp/tdc_clearance/model_and_evals
```

To use the model later for scoring, run:

```bash
# assuming you copied the file tdc_microsome_test.csv in this repo to /tmp/tdc_clearance
optimol_score --config-path /tmp/tdc_clearance/model_and_evals/.hydra  \
        --config-name config.yaml \
        ckpt_path=/tmp/tdc_clearance/model_and_evals \
        task_name=eval \
        data.test_data=/tmp/tdc_clearance/tdc_microsome_test.csv \
        logger=null \
        data.ligand_field=smiles \
        data.ligand_id_field=name \
        hydra.run.dir=/tmp/tdc_clearance/predictions
```