def execute_galaxy_operator(pipeline, dataset, action_type):

    parts = action_type.split('-&-')
    operator = parts[0]
    if operator not in ('by_facet', 'by_neighbors', 'by_superset', 'by_distribution'):
        raise ValueError('Unknown Galaxy operator: ' + operator)
    components = dataset.predicate.components
    if operator in ('by_facet', 'by_neighbors'):
        if len(parts) != 2 or not parts[1]:
            raise ValueError('Missing Galaxy operator dimension: ' + action_type)
        dimension = parts[1]
        constrained = any(item.attribute == dimension for item in components)
        if (operator == 'by_facet' and constrained) or (operator == 'by_neighbors' and not constrained):
            return []
        if operator == 'by_facet':
            result = pipeline.by_facet(dataset, [dimension], pipeline.discrete_categories_count)
        else:
            result = pipeline.by_neighbors(dataset, [dimension])
    else:
        if len(components) <= 1:
            return []
        result = getattr(pipeline, operator)(dataset)
    if result is None:
        return []
    datasets = result if isinstance(result, list) else [result]
    return [item for item in datasets
            if item is not None and item.data is not None and not item.data.empty]


def complete_training(ray, parameter_server, workers, tasks):

    try:
        ray.get(tasks)
        ray.get(parameter_server.save_final_models.remote())
    except BaseException:
        for actor in list(workers) + [parameter_server]:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:


                import logging
                logging.getLogger(__name__).exception('Could not stop Galaxy actor')
        raise


def _indexed_worker(payload):
    worker, index, task = payload
    return index, worker(task)


def run_worker_batch(worker, tasks, processes):

    if len(tasks) == 1:
        return [worker(tasks[0])]
    import multiprocessing as mp
    results = [None] * len(tasks)
    with mp.Pool(processes=processes) as pool:
        payloads = [(worker, index, task) for index, task in enumerate(tasks)]


        for index, result in pool.imap_unordered(_indexed_worker, payloads):
            results[index] = result
    return results
