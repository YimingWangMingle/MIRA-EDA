from revision_support.training_runtime import configure_training_threads

configure_training_threads()

from baselines.greedy_eda.galaxy import parse_args, run


if __name__ == "__main__":
    run(parse_args())
