# Usage

This folder contains scripts to reproduce the results from our paper.



## Experiments

The folder `exps/` contains scripts that can be used to reproduce the experiments from our paper. To this end, run 

```bash
python -m scripts.exps.sample_efficiency prop
```

in the root folder, where `prop` is `YM_Density`, `YM_Density_RI`, `LogViscosity_Tg`, or `LogViscosity_Tg_t`. 
This starts `MAX_N_PROCESSES` (default 80) separate processes which are run in parallel. Each process requires ca. 480 MB in memory.
The results are saved with a specified identifier (default is the current timestamp) to `glassmtl/results/`.



## Evaluation

The folder `evals/` contains scripts that can be used to generate evaluation data. To reproduce the data from our paper, run

```bash
python -m scripts.evals.eval_gen
```

in the root folder. Adapt the identifier `ident` in the script accordingly for evaluations of own experiments.



## Results

All data is saved to `glassmtl/results/`. Please see [here](results/README.md) for further information.
