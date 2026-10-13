import json
import random
from os import listdir
from os.path import isfile, join


class TargetSetGenerator:
    @staticmethod
    def _iter_target_json_files(target_dir="./rl/targets/", include_generated=False):
        for name in sorted(listdir(target_dir)):
            path = join(target_dir, name)
            if not isfile(path) or not name.endswith(".json"):
                continue
            if not include_generated and name.startswith("fixed_"):
                continue
            yield path

    @staticmethod
    def get_diverse_target_set(number_of_samples=10, seed=None, target_dir="./rl/targets/"):
        rng = random.Random(seed) if seed is not None else random
        initial_target_items = []
        for path in TargetSetGenerator._iter_target_json_files(target_dir=target_dir):
            with open(path) as f:
                items = json.load(f)
                if len(items) > number_of_samples:
                    initial_target_items += rng.choices(items, k=number_of_samples)
                else:
                    initial_target_items += items
        return set(initial_target_items)

    @staticmethod
    def save_diverse_target_set(output_name, number_of_samples=100, seed=0, target_dir="./rl/targets/"):
        target_items = TargetSetGenerator.get_diverse_target_set(
            number_of_samples=number_of_samples,
            seed=seed,
            target_dir=target_dir,
        )
        output_path = join(target_dir, f"{output_name}.json")
        with open(output_path, "w") as f:
            json.dump(sorted(map(int, target_items)), f, indent=1)
        return output_path, len(target_items)

    @staticmethod
    def get_concentrated_target_set(seed=None):
        rng = random.Random(seed) if seed is not None else random
        target_files = list(TargetSetGenerator._iter_target_json_files())
        while True:
            target_file = rng.choice(target_files)
            with open(target_file) as f:
                items = json.load(f)
            if 50 < len(items) < 2000:
                print(target_file)
                break
        return set(items)
