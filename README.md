# MIRA

We provide our code and data for Galaxy, Covertype, and the ATENA benchmark (four Cyber and four Flights datasets). We include the fixed targets, indexes, and preprocessing caches needed to run the experiments. Results, trained models, and fitted PCA projections are generated during a run.

## Setup

We use Linux and Python 3.9 for Galaxy and Covertype. We split the original Galaxy dataset into two files. From the repository root, we restore it once:

```bash
if [ ! -f app/data/sdss/galaxies.csv ]; then
  cat app/data/sdss/galaxies.csv.gz.part* | gzip -dc > app/data/sdss/galaxies.csv.tmp &&
    mv app/data/sdss/galaxies.csv.tmp app/data/sdss/galaxies.csv
fi
```

We then install the dependencies and check the package:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B verify.py --runtime
```

The check covers dependencies, file hashes, data, caches, command arguments, and model updates. No data download or preprocessing is required.

## Galaxy

We first check a short MIRA run, then run all methods:

```bash
python -B run.py --dataset galaxy --method mira --seeds 1 --smoke --execute
python -B run.py --dataset galaxy --method all --execute
```

## Covertype

We use the same runner and training budget:

```bash
python -B run.py --dataset covertype --method mira --seeds 1 --smoke --execute
python -B run.py --dataset covertype --method all --execute
```

For each dataset, we run MIRA, seven comparison methods, and seven replacement ablations on fixed target seeds 1, 2, and 3: 45 runs per dataset. Each full run uses 1,000 episodes of 250 steps with 14 workers and one numerical thread per process. The target seeds define different target tasks. Asynchronous training is not bitwise deterministic.

We use `--method baselines` or `--method ablations` to run either group, or a method ID from [protocol.json](protocol.json) to run one method. Omitting `--execute` prints the commands. PCA uses the first 16 episodes for collection and fitting, then trains for the remaining 984 episodes within the same budget.

We write full runs to `runs/`, short checks to `smoke_runs/`, and resource measurements to the corresponding `cost/` directory. `--output /path/to/new_runs` changes the output location. Each run requires a fresh output directory.

## ATENA

We compare MIRA, Official ATENA, PureA3C, DORA, Greedy, and Random across all eight datasets. We do not run replacement ablations on this benchmark.

From the repository root, we set up the TensorFlow environment, check one dataset, and run our methods:

```bash
cd ATENA_dataset
PYTHON_BIN=python3.9 bash scripts/setup_venv.sh
source .venv/bin/activate
python MIRA/run.py --schema cyber --dataset_number 1 --workers 2 --steps 24 --eval_interval 1 --output_dir results/smoke
SEED=0 WORKERS=16 STEPS=1000000 bash MIRA/scripts/run_all.sh
SEED=0 WORKERS=16 STEPS=1000000 bash scripts/run_baselines.sh
```

We retain the official ATENA implementation and data. Its original dependency stack needs a separate Python 3.8 environment. Continuing from `ATENA_dataset/`:

```bash
PYTHON_BIN=python3.8 bash scripts/setup_official_atena_venv.sh
source ATENA-A-EDA/atena-basic/.venv-official-atena/bin/activate
SEED=0 NUM_ENVS=16 STEPS=1000000 bash scripts/run_official_atena_template.sh
SEED=0 CHECKPOINT_STEP=1000000_finish bash scripts/evaluate_official_atena_batch.sh
```

We then return to our environment and compute the final corpus metrics and cost summaries:

```bash
source .venv/bin/activate
python scripts/recalculate_baselines.py --results_dir results --seed 0
python scripts/summarize_costs.py --root results --seed 0
```

We use seed 0 by default and record actual interaction counts after whole-episode rounding. Each episode has 12 actions. Random reports the mean of 16 evaluation sessions. MIRA uses policy weight averaging from 40% of training onward. Results and checkpoints are written under `ATENA_dataset/results/`.

We record the fixed settings in [protocol.json](protocol.json) and file hashes in [manifest.json](manifest.json). The shared MIRA core is in `revision_support/`; dataset adapters remain in their respective directories. This release covers the algorithm core; shared query caching and batch scheduling are not implemented.
