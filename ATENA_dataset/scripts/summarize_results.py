import argparse
import csv
import json
import math
import numbers
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from atena_common.costs import average_random_rows

IDENTITY_FIELDS = ("method", "schema", "dataset", "seed", "steps")
OFFICIAL_METRIC_FIELDS = (
    "Precision",
    "T-BLEU-1",
    "T-BLEU-2",
    "T-BLEU-3",
    "EDA-Sim",
)


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_row(row, source):
    if not isinstance(row, dict):
        raise ValueError(f"{source} must contain an object row")
    for field in ("method", "schema"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"{source} {field} must be a non-empty string")
    for field, minimum in (("dataset", 1), ("seed", 0), ("steps", 0)):
        value = row.get(field)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{source} {field} must be an integer >= {minimum}")
    for field in OFFICIAL_METRIC_FIELDS:
        if field not in row:
            raise ValueError(f"{source} missing official metric: {field}")
        _finite_number(row[field], f"{source} {field}")
    if "policy_source" in row and row["policy_source"] not in ("online", "swa"):
        raise ValueError(f"{source} policy_source must be online or swa")
    for field, value in row.items():
        if field in {"method", "schema", "policy_source"}:
            continue
        if isinstance(value, bool):
            raise ValueError(f"{source} {field} must not be boolean")
        if isinstance(value, numbers.Real):
            _finite_number(value, f"{source} {field}")
    return row


def _strict_json(path):
    try:
        return json.loads(
            Path(path).read_bytes().decode("utf-8-sig"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON: {path}") from exc


def _parse_csv_int(value, name, minimum):
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise ValueError(f"{name} must be a canonical integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _parse_csv_row(raw, fieldnames, row_number, path):
    if None in raw or any(raw.get(field) is None for field in fieldnames):
        raise ValueError(f"{path} row {row_number} has missing or extra cells")
    row = {}
    for field in fieldnames:
        value = raw[field]
        if field in {"method", "schema", "policy_source"}:
            if not value:
                raise ValueError(f"{path} row {row_number} {field} is empty")
            row[field] = value
        elif field == "dataset":
            row[field] = _parse_csv_int(value, f"{path} row {row_number} dataset", 1)
        elif field == "seed":
            row[field] = _parse_csv_int(value, f"{path} row {row_number} seed", 0)
        elif field == "steps":
            row[field] = _parse_csv_int(value, f"{path} row {row_number} steps", 0)
        else:
            try:
                row[field] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path} row {row_number} {field} must be numeric"
                ) from exc
    return _validate_row(row, f"{path} row {row_number}")


def _load_metrics_csv(path):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fieldnames = reader.fieldnames
            if (
                not fieldnames
                or len(fieldnames) != len(set(fieldnames))
                or not set(IDENTITY_FIELDS + OFFICIAL_METRIC_FIELDS).issubset(fieldnames)
            ):
                raise ValueError(f"{path} has an unsupported header")
            raw_rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"malformed CSV: {path}") from exc
    if not raw_rows:
        raise ValueError(f"{path} must contain at least one row")
    rows = [
        _parse_csv_row(raw, fieldnames, row_number, path)
        for row_number, raw in enumerate(raw_rows, start=2)
    ]
    identity = tuple(rows[0][field] for field in ("method", "schema", "dataset", "seed"))
    for row in rows[1:]:
        if tuple(row[field] for field in ("method", "schema", "dataset", "seed")) != identity:
            raise ValueError(f"{path} contains mixed run identities")
    return average_random_rows(rows) if rows[0]["method"] == "random" else rows[-1]


def main():
    parser = argparse.ArgumentParser(description="Summarize formal evaluation metric files.")
    parser.add_argument("--root", default="results")
    parser.add_argument("--output", default="results/atena_baselines_summary.csv")
    parser.add_argument(
        "--seed",
        default="0",
        help="Seed directory to summarize, e.g. 0 for seed0. Use 'all' only for optional robustness runs.",
    )
    parser.add_argument(
        "--row-source",
        dest="row_source",
        choices=["last", "final_json"],
        default="last",
        help="Use the last eval_metrics.csv row or final_metrics.json.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if str(args.seed).lower() == "all":
        seed_pattern = "seed*"
    else:
        seed_pattern = f"seed{int(args.seed)}"

    rows = []
    for seed_dir in sorted(root.glob(f"*/*/{seed_pattern}")):
        if not seed_dir.is_dir():
            continue
        row = load_run_row(seed_dir, args.row_source)
        if row:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No result rows found under {root} for {seed_pattern}")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} final rows to {output}")


def load_run_row(seed_dir: Path, row_source: str = "last"):
    seed_dir = Path(seed_dir)
    if row_source == "final_json":
        final_path = seed_dir / "final_metrics.json"
        if not final_path.exists():
            return None
        return _validate_row(_strict_json(final_path), str(final_path))
    if row_source != "last":
        raise ValueError(f"unsupported row source: {row_source}")

    metrics_path = seed_dir / "eval_metrics.csv"
    if not metrics_path.exists():
        return None
    return _load_metrics_csv(metrics_path)


if __name__ == "__main__":
    main()
