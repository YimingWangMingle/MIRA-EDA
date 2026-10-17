from atena.evaluation.metrics import (
    EvalInstance,
    DisplaysTreeBleuMetric,
    PrecisionMetric,
    NormalizedDisplaysEdaSimMetric, get_dataframe_all_eval_metrics
)

from atena.evaluation.references.cyber.dataset1 import cyber1_references
from atena.simulation.actions_simulator import ActionsSimulator
from atena.simulation.dataset import (
    Dataset,
    DatasetMeta,
    SchemaName,
)
from atena.simulation.utils import random_action_generator

if __name__ == "__main__":


    eval_instances = []

    for schema in SchemaName:
        for dataset_name in schema.dataset_names:
            eval_dataset_meta = DatasetMeta(schema, dataset_name=dataset_name)
            eval_instance = EvalInstance(
                eval_dataset_meta,
                actions_lst=[random_action_generator(Dataset(eval_dataset_meta)) for _ in range(12)]
            )
            eval_instances.append(eval_instance)

    eval_metrics = [
        PrecisionMetric(eval_instances),
        DisplaysTreeBleuMetric(1, eval_instances),
        DisplaysTreeBleuMetric(2, eval_instances),
        DisplaysTreeBleuMetric(3, eval_instances),
        NormalizedDisplaysEdaSimMetric(eval_instances)
    ]


    get_dataframe_all_eval_metrics(eval_instances)
