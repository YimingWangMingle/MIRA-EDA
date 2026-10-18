from revision_support.training_runtime import configure_training_threads

configure_training_threads()

import ray

from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from baselines.pure_a3c.A3C_2_actors.A3C import Agent, build_parser


from revision_support.query_service import shared_query_run

@shared_query_run
def execute(args):
    ray.init(ignore_reinit_error=True)

    env_name = "pipeline"
    data_folder = "./app/data/"
    pipeline = PipelineWithPrecalculatedSets(
        "sdss",
        ["galaxies"],
        data_folder=data_folder,
        discrete_categories_count=10,
        min_set_size=10,
        exploration_columns=[
            "galaxies.u",
            "galaxies.g",
            "galaxies.r",
            "galaxies.i",
            "galaxies.z",
            "galaxies.petroRad_r",
            "galaxies.redshift",
        ],
    )

    agent = Agent(env_name, pipeline, args=args)
    agent.train(args.episodes)
    ray.shutdown()


def main():
    from revision_support.query_config import configure_query_processing
    args = build_parser().parse_args()
    configure_query_processing(args, method="pure_a3c")
    execute(args)

if __name__ == "__main__":
    main()
