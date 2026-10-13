import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from covertype_rl.full_a3c import build_parser, run


def main():
    parser = build_parser()
    parser.set_defaults(baseline="atena_no_ext")
    args = parser.parse_args()
    paths = run(args)
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
