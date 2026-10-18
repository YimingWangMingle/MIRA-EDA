# MIRA

We provide our code and data for Galaxy, Covertype, and the ATENA benchmark (four Cyber and four Flights datasets). We include the fixed targets, indexes, and preprocessing caches needed to run the experiments. Results, trained models, and fitted PCA projections are generated during a run.

full version of our code will be released after acceptance.

## Setup

We use Linux and Python 3.9 for Galaxy and Covertype. The Galaxy CSV is stored as lossless gzip parts to fit GitHub file limits. From the repository root, restore it before running validation or training:

```bash
if [ ! -f app/data/sdss/galaxies.csv ]; then
  cat app/data/sdss/galaxies.csv.gz.part* | gzip -dc > app/data/sdss/galaxies.csv.tmp &&
    mv app/data/sdss/galaxies.csv.tmp app/data/sdss/galaxies.csv
fi
```

Install the dependencies and check the package from its root directory:

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

For each dataset, we run MIRA, seven comparison methods, and ten replacement ablations on fixed target seeds 1, 2, and 3: 54 runs per dataset. Each full run uses 1,000 episodes of 250 steps with 14 workers and one numerical thread per process. The target seeds define different target tasks. Asynchronous training is not bitwise deterministic.

We use `--method baselines` or `--method ablations` to run either group, or one of the method IDs listed below to run one method. Omitting `--execute` prints the commands. PCA uses the first 16 episodes for collection and fitting, then trains for the remaining 984 episodes within the same budget.

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

We use seed 0 by default and record actual interaction counts after whole-episode rounding. Each episode has 12 actions. Random reports the mean of 16 evaluation sessions. Results and checkpoints are written under `ATENA_dataset/results/`.

We record file hashes in `manifest.json`. The shared MIRA core is in `revision_support/`; dataset adapters remain in their respective directories. MIRA uses the same environment observations as the corresponding baselines and conditions its policy on an episode-level direction. ATENA evaluates and saves the current policy after training updates.

## Galaxy and Covertype method IDs

| Method ID | Training reward |
|---|---|
| `mira` | `2E + 0.5B` |
| `mira_no_ext` | `0.5B` |
| `pure_a3c` | `2E` |
| `dora` | `0.75E + 0.25counter` |
| `atena` | `2E + 0.1I + 0.1C + 0.1D` |
| `atena_no_ext` | `0.1I + 0.1C + 0.1D` |
| `greedy_eda` | Non-learning selection policy |
| `random` | Non-learning random policy |

E denotes the external task reward, B the directional exploration bonus, I interestingness, C coherency, and D diversity. The representation learner uses the external task reward for `mira` and a zero task-reward input for `mira_no_ext`; both retain analytical grounding and the latent dynamics objective.

The replacement method IDs are `encoder_stats_pca`, `encoder_vae`, `encoder_frozen_random`, `reward_rnd`, `reward_count`, `reward_knn`, and `reward_random`. Each replaces one MIRA component. All comparison methods and these encoder/reward ablations enable result reuse, heuristic pruning, and shared execution by default.

We add three MIRA query ablations, each disabling only the named component while keeping the default encoder and reward:

| Method ID | Disabled component |
|---|---|
| `query_no_result_reuse` | Completed-result reuse and identical pending-request coalescing |
| `query_no_heuristic_pruning` | Heuristic candidate pruning |
| `query_no_shared_execution` | Shared intermediate execution |

DORA and ATENA retain all three optimizations; All Covertype RL variants use a common 202-slot capacity, with the default heuristic retaining at most ten successors and the no-pruning variant retaining all legal successors. This increases memory use and requires fresh training for all variants; old ten-slot checkpoints and timing results are not comparable.

```bash
for dataset in galaxy covertype; do
  python -B run.py --dataset "$dataset" --method query_no_result_reuse --execute
  python -B run.py --dataset "$dataset" --method query_no_heuristic_pruning --execute
  python -B run.py --dataset "$dataset" --method query_no_shared_execution --execute
done
```

## Validation

`python -B verify.py` checks source, command arguments, file hashes, and bundled data. `python -B verify.py --runtime` also checks the pinned Linux runtime and model updates. To run the regression checks, install `pytest` and run `python -B -m pytest tests -q`.

LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
