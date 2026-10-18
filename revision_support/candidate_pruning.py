import numpy as np
from .query_config import heuristic_pruning_enabled


def galaxy_candidate_mask(dataset, action_types, exact_mask):
    exact = np.asarray(exact_mask, dtype=np.float64)
    if not heuristic_pruning_enabled():
        return exact
    reduced = exact.copy()


    columns = [name.split('-&-', 1)[1] for name, valid in zip(action_types, exact)
               if valid and name.startswith('by_facet-&-')]
    if columns:
        from .query_backends import galaxy_summary
        summary = galaxy_summary(dataset, columns)
        for i, name in enumerate(action_types):
            if exact[i] and name.startswith('by_facet-&-'):
                if summary[name.split('-&-', 1)[1]]['distinct'] <= 1:
                    reduced[i] = 0.

    return reduced if reduced.any() else exact
def sample_family_action(action_groups, rng, valid_ids=None):
    allowed = None if valid_ids is None else set(map(int, valid_ids))
    groups = {name: [i for i in ids if allowed is None or i in allowed]
              for name, ids in action_groups.items()}
    families = [name for name, ids in groups.items() if ids]
    if not families:
        raise ValueError('No legal operator action is available')
    family = str(rng.choice(families))
    return int(rng.choice(groups[family]))
