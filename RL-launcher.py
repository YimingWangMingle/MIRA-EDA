from revision_support.training_runtime import configure_training_threads

configure_training_threads()

import ray
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from rl.A3C_2_actors.A3C import Agent

def main():
    # 初始化 Ray，利用底层 C++ 引擎接管多进程调度与共享内存
    ray.init(ignore_reinit_error=True)
    
    env_name = 'pipeline'
    data_folder = "./app/data/"
    pipeline = PipelineWithPrecalculatedSets(
        "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])
    
    agent = Agent(env_name, pipeline)
    agent.train()
    
    ray.shutdown()

if __name__ == "__main__":
    main()
