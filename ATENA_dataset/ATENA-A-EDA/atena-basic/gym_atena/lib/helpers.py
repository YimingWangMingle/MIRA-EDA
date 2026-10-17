from collections import namedtuple, deque
from enum import Enum
import math
from functools import lru_cache

import numpy as np
import operator

from scipy.stats import entropy

from Utilities.Collections.Counter_Without_Nans import CounterWithoutNanKeys


FilteringTuple = namedtuple('FilteringTuple', ["field", "term", "condition"])
AggregationTuple = namedtuple('AggregationTuple', ["field", "type"])


class GetItemByStr(tuple):
    __slots__ = ()

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return self.__getattribute__(attr)
        else:
            return tuple.__getitem__(self, attr)


class AggAttrTuple(namedtuple('AggAttrTuple', ["column", "grouped_or_aggregated_state"]), GetItemByStr):

    __slots__ = ()

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)


class GranLayerTuple(namedtuple('GranLayerTuple', ["group_attrs", "agg_attrs",
                                                   "inverse_ngroups", "site_std", "inverse_size_mean"]), GetItemByStr):

    __slots__ = ()

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)


class ColumnDataLayerTuple(namedtuple('ColumnDataLayerTuple', ["unique", "nulls", "entropy"]), GetItemByStr):

    __slots__ = ()

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)


class DisplayTuple(namedtuple('DisplayTuple', ["data_layer", "granularity_layer"]), GetItemByStr):

    __slots__ = ()

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)


class EnvStateTuple(namedtuple('EnvStateTuple', ["filtering", "grouping", "aggregations"]), GetItemByStr):

    __slots__ = ()

    def reset_filtering(self):
        return self._create_state_tuple(filtering=tuple(),
                                        grouping=self.grouping,
                                        aggregations=self.aggregations)

    def reset_grouping_and_aggregations(self):
        return self._create_state_tuple(filtering=self.filtering,
                                        grouping=tuple(),
                                        aggregations=tuple())

    def append_filtering(self, elem):
        field_lst = self._append_to_field(elem, "filtering")
        return self._create_state_tuple(filtering=field_lst,
                                        grouping=self.grouping,
                                        aggregations=self.aggregations)

    def append_grouping(self, elem):
        field_lst = self._append_to_field(elem, "grouping")
        return self._create_state_tuple(filtering=self.filtering,
                                        grouping=field_lst,
                                        aggregations=self.aggregations)

    def append_aggregations(self, elem):
        field_lst = self._append_to_field(elem, "aggregations")
        return self._create_state_tuple(filtering=self.filtering,
                                        grouping=self.grouping,
                                        aggregations=field_lst)

    def _append_to_field(self, elem, field):
        field_lst = list(self[field])
        field_lst.append(elem)
        return field_lst

    def __getitem__(self, attr):
        return GetItemByStr.__getitem__(self, attr)

    @classmethod
    def _create_state_tuple(cls, filtering, grouping, aggregations):
        return cls(
            filtering=tuple(filtering),
            grouping=tuple(grouping),
            aggregations=tuple(aggregations),
        )

    @classmethod
    def create_empty_state(cls):
        return cls._create_state_tuple((), (), ())


empty_env_state = EnvStateTuple.create_empty_state()


def hack_min(pd_series):
    return np.min(pd_series.dropna())


def hack_max(pd_series):
    return np.max(pd_series.dropna())


KL_DIV_EPSILON = 0.05


INVERSE_OPERATOR_TYPE_LOOKUP = {
        "back": 0,
        "filter": 1,
        "group": 2,
        }

OPERATOR_TYPE_LOOKUP = {
        0: "back",
        1: "filter",
        2: "group",
        }


INT_OPERATOR_MAP_ATENA = {
    0: operator.eq,
    1: operator.gt,
    2: operator.ge,
    3: operator.lt,
    4: operator.le,
    5: operator.ne,
    6: None,
    7: None,
    8: None,
}


INT_OPERATOR_MAP_ATENA = {
    0: operator.eq,
    1: operator.eq,
    2: operator.eq,
    3: operator.ne,
    4: operator.ne,
    5: operator.ne,
    6: None,
    7: None,
    8: None,
}

INT_OPERATOR_MAP_ATENA_STR = {
    0: "eq",
    1: "eq",
    2: "eq",
    3: "ne",
    4: "ne",
    5: "ne",
    6: "contains",
    7: "contains",
    8: "contains",
}

INT_OPERATOR_MAP_ATENA_PRETTY_STR = {
    0: "EQUAL",
    1: "EQUAL",
    2: "EQUAL",
    3: "NOT EQUAL",
    4: "NOT EQUAL",
    5: "NOT EQUAL",
    6: "CONTAINS",
    7: "CONTAINS",
    8: "CONTAINS",
}

PRETTY_STR_MAP_ATENA_INT_OPERATOR = {
    "EQUAL": 0,
    "NOT EQUAL": 3,
    "CONTAINS": 6,
}


INT_OPERATOR_MAP_REACT = {
    8: operator.eq,
    32: operator.gt,
    64: operator.ge,
    128: operator.lt,
    256: operator.le,
    512: operator.ne,
}

INT_OPERATOR_MAP_REACT_TO_ATENA = {
    2: 7,
    4: 8,
    8: 0,
    16: 6,
    32: 1,
    64: 2,
    128: 3,
    256: 4,
    512: 5,
}


AGG_MAP_ATENA = {
    0: np.sum,
    1: len ,
    2: hack_min,
    3: hack_max,
    4: np.mean
}

AGG_MAP_ATENA = {
    0: len,
}

AGG_MAP_ATENA_STR = {
    0: 'COUNT',
}

INVERSE_AGG_MAP_ATENA = {
    'sum': 0,
    'count': 1,
    'min': 2,
    'max': 3,
    'avg': 4

}

INVERSE_AGG_MAP_ATENA = {
    'count': 0,
}


AGG_MAP_react = {
    'sum': np.sum,
    'count': len,
    'min': hack_min,
    'max': hack_max,
    'avg': np.mean
}


class ActionVectorEntry(Enum):
    ACTION_TYPE = 0
    COL_ID = 1
    FILTER_OP = 2
    FILTER_TERM = 3
    AGG_COL_ID = 4
    AGG_FUNC = 5


def lst_of_actions_to_tuple(actions_lst):

    return tuple([tuple(action) for action in actions_lst])


def get_aggregate_attributes(state):
    agg_attributes = set()
    aggs_state = state["aggregations"]
    for agg_tple in aggs_state:
        agg_attributes.add(agg_tple.field)
    return list(agg_attributes)


@lru_cache(maxsize=500)
def normalized_sigmoid_fkt(a, b, x):

    return 1/(1+math.exp(b*(x-a)))
