from collections import namedtuple
from typing import NamedTuple

from atena.simulation.actions import AggregationFunction

FilteringTuple = namedtuple('FilteringTuple', ["column", "term", "operator"])


class AggregationTuple(NamedTuple):
    column: str
    aggregation_function: AggregationFunction


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

    def is_grouped(self):
        return self.grouping

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
