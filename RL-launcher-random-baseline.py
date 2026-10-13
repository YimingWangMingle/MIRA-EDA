from revision_support.training_runtime import configure_training_threads

configure_training_threads()

from baselines.random.random_baseline import parse_args, run


if __name__ == "__main__":
    run(parse_args())
