"""MIRA-only descriptors. No target membership, task reward or evaluation input."""
import numpy as np


class GalaxyDescriptor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.names = ['log_cardinality', 'selectivity']
        self.statistic_indices = [0, 1]
        self.categories = {}
        for column in pipeline.exploration_columns:
            categories = list(map(str, pipeline.ordered_dimensions[column]))
            self.categories[column] = categories
            offset = len(self.names)
            self.names += [column+suffix for suffix in ('_predicate_present', '_predicate_value', '_distinct', '_entropy')]
            self.names += [column+'_p_'+str(i) for i in range(len(categories)+1)]
            self.statistic_indices += list(range(offset+2, len(self.names)))

    def encode(self, dataset):
        data = dataset.data
        n, total = len(data), max(1, len(self.pipeline.initial_collection))
        values = [np.log1p(n)/np.log1p(total), n/total]
        for column, categories in self.categories.items():
            predicate = next((p for p in dataset.predicate.components if p.attribute == column), None)
            term = str(predicate.value) if predicate is not None else None
            if term is not None and term not in categories:
                raise ValueError('Predicate value is outside the declared Galaxy schema')
            values += [float(predicate is not None), categories.index(term)/max(len(categories)-1, 1) if term is not None else 0.]
            counts = data[column].astype(str).value_counts(dropna=False)
            probabilities = np.array([float(counts.get(v,0)) for v in categories]+[
                float(sum(count for key,count in counts.items() if key not in categories))], dtype=np.float64)/max(n,1)
            positive = probabilities[probabilities > 0]
            entropy = -np.sum(positive*np.log(positive))/np.log(max(2,len(probabilities)))
            values += [data[column].nunique()/max(n,1), entropy]
            values.extend(probabilities.tolist())
        return np.asarray(values, dtype=np.float32)


def covertype_region(universe, set_id):
    total = max(1, universe.size_for_set(0))
    size = universe.size_for_set(set_id)
    return np.concatenate([universe.state_for_set(set_id),
        [np.log1p(size)/np.log1p(total), size/total]]).astype(np.float32)


def action_history(actions, action_dim, horizon):
    """Exact ordered parameterized-action identities, including BACK and no-ops."""
    result = np.zeros((int(horizon), int(action_dim)), dtype=np.float32)
    if len(actions) > horizon:
        raise ValueError('History exceeds declared episode horizon')
    for step, action in enumerate(actions):
        if not 0 <= int(action) < action_dim:
            raise ValueError('History action outside declared schema')
        result[step, int(action)] = 1.
    return result.reshape(-1)
