from contextvars import ContextVar
import hashlib
import pickle

import numpy as np
from scipy.stats import entropy

GALAXY_BATCH = ContextVar('galaxy_query_batch', default=None)


def fingerprint(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def galaxy_summary(dataset, columns):
    key = fingerprint(dataset.data.loc[:, columns])
    cached = getattr(dataset, '_query_summary', None)
    import os
    reuse = os.environ.get('MIRA_RESULT_REUSE', '1') == '1'
    if reuse and cached is not None and cached[0] == key:
        return cached[1]
    summary = {}
    for column in columns:
        counts = dataset.data[column].value_counts()
        summary[column] = dict(counts=counts, distinct=int((counts > 0).sum()),
            entropy=float(entropy(counts) if not counts.empty and counts.sum() > 0 else 0.))
    if reuse:
        dataset._query_summary = (key, summary)
    return summary


class GalaxyPredicateBatch:
    def __init__(self):
        self.prefixes = {}
        self.shared_intermediates = 0

    def filter(self, pipeline, data, predicate):
        source = fingerprint(data)
        prefix = []
        for item in predicate.components:
            value = item.value
            if item.is_multiple_line_string:
                if type(value) is str:
                    value = value.replace('\n', '\\n')
            elif item.is_category:
                index = pipeline.ordered_dimensions[item.attribute].index(str(value))
                value = pipeline.initial_collection[item.attribute].dtype.categories[index]
            prefix.append((item.attribute, value))
            key = (source, pickle.dumps(prefix, protocol=4))
            if key in self.prefixes:
                data = self.prefixes[key]
                self.shared_intermediates += 1
            else:
                data = data[data[item.attribute] == value]
                self.prefixes[key] = data
        return data.copy()


class GalaxyBackend:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def group_key(self, request):
        action, data, _, _, _, _ = request

        return (action.split('-&-')[0], fingerprint(data), tuple(data.columns))

    def evaluate_many(self, requests, share=True):
        from app.pipelines.dataset import Dataset
        from .galaxy_execution import execute_galaxy_operator
        batch = GalaxyPredicateBatch() if share else None
        token = GALAXY_BATCH.set(batch)
        results = []
        try:
            for action, data, predicate, joins, tables, set_id in requests:
                dataset = Dataset(set_id=set_id, data=data.copy(), predicate=predicate,
                                  joins=joins, tables=tables)
                result = execute_galaxy_operator(self.pipeline, dataset, action, _local=True)
                for item in result:
                    galaxy_summary(item, self.pipeline.exploration_columns)
                results.append(result)
            return results, batch.shared_intermediates if batch else 0
        finally:
            GALAXY_BATCH.reset(token)


class DisplayBackend:
    def __init__(self, calculator):
        self.calculator = calculator

    def group_key(self, state):

        return pickle.dumps(state['filtering'], protocol=4)

    def evaluate_many(self, states, share=True):
        from atena.simulation.state import DisplayTuple
        calculator = self.calculator
        parents, results, reused = {}, [], 0
        for state in states:
            key = self.group_key(state)
            if share and key in parents:
                filtered, data_layer = parents[key]
                reused += 1
            else:
                filtered = calculator.get_filtered_df(state['filtering'])
                data_layer = calculator.calc_data_layer(filtered)
                if share:
                    parents[key] = (filtered, data_layer)
            grouped, aggregated = calculator.get_groupby_df(filtered, state['grouping'], state['aggregations'])
            display = DisplayTuple(data_layer=data_layer,
                granularity_layer=calculator.calc_gran_layer(grouped, aggregated))
            results.append((display, (filtered, aggregated)))
        return results, reused


class NativeDisplayBackend:
    def __init__(self, properties, dataframe):
        from types import SimpleNamespace
        from gym_atena.envs.env_properties import EnvProp
        self.properties = object.__new__(EnvProp)
        self.properties.env_dataset_prop = SimpleNamespace(
            KEYS=properties.env_dataset_prop.KEYS,
            NUMERIC_KEYS=properties.env_dataset_prop.NUMERIC_KEYS)
        self.properties.INT_OPERATOR_MAP_ATENA = properties.INT_OPERATOR_MAP_ATENA
        self.properties.AGG_MAP_ATENA = properties.AGG_MAP_ATENA
        self.dataframe = dataframe

    def group_key(self, state):
        return pickle.dumps(state['filtering'], protocol=4)

    def evaluate_many(self, states, share=True):
        parents, results, reused = {}, [], 0
        prop = self.properties
        for state in states:
            key = self.group_key(state)
            if share and key in parents:
                filtered, data_layer = parents[key]
                reused += 1
            else:
                filtered = prop.get_filtered_df(self.dataframe, state['filtering'])
                data_layer = prop.calc_data_layer(filtered)
                if share:
                    parents[key] = (filtered, data_layer)
            grouped, aggregated = prop.get_groupby_df(filtered, state['grouping'], state['aggregations'])
            results.append((filtered, aggregated, data_layer, prop.calc_gran_layer(grouped, aggregated)))
        return results, reused


class CovertypeBackend:
    def __init__(self, data):
        from types import SimpleNamespace
        self.data = SimpleNamespace(**{key: np.asarray(getattr(data, key)).copy()
            for key in ('continuous_bins', 'continuous_norm', 'cover_type', 'wilderness', 'soil')})
        for array in vars(self.data).values():
            array.flags.writeable = False

    def _terms(self, request):
        values = tuple(int(v) for v in request)
        n_cont = self.data.continuous_bins.shape[1]
        if len(values) != n_cont + 3:
            raise ValueError('Covertype query constraint dimension mismatch')
        return tuple((i, v) for i, v in enumerate(values)
                     if (v >= 0 if i < n_cont else v > 0))

    def group_key(self, request):

        return self._terms(request)[:1]

    def evaluate_many(self, requests, share=True):
        from covertype_rl.fixed_sets import _state_for_indices
        data = self.data
        n_cont = data.continuous_bins.shape[1]
        categorical = (data.cover_type, data.wilderness, data.soil)
        prefixes, results, reused = {}, [], 0
        root = np.arange(len(data.continuous_bins), dtype=np.int64)
        for request in requests:
            indices, prefix = root, ()
            for column, value in self._terms(request):
                prefix += ((column, value),)
                if share and prefix in prefixes:
                    indices = prefixes[prefix]
                    reused += 1
                else:
                    values = data.continuous_bins[:, column] if column < n_cont else categorical[column-n_cont]
                    indices = indices[values[indices] == value]
                    if share:
                        prefixes[prefix] = indices
            results.append(dict(indices=indices.copy(), set_size=len(indices),
                                state=_state_for_indices(data, request, indices)))
        return results, reused
