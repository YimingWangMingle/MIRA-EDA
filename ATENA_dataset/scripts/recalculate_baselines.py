import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "ATENA-A-EDA" / "benchmark"
for path in (ROOT, BENCHMARK, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from recalculate_baseline_corpus_metrics.compat import enable_legacy_pandas

enable_legacy_pandas()

from recalculate_baseline_corpus_metrics.application import recalculate_all


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    _, summary, _, paths = recalculate_all(args.results_dir, seed=args.seed)
    for row in summary:
        values = " ".join(
            f"{name}={row[name]:.9f}"
            for name in ("Precision", "T-BLEU-1", "T-BLEU-2", "T-BLEU-3", "EDA-Sim")
        )
        print(f"{row['method']}: {values}")
    print("Outputs:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
