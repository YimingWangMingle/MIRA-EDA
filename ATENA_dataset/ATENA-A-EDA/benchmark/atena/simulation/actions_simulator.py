from typing import List

import pandas as pd

from atena.simulation.actions import (
    AbstractAction,
    ActionType,
    FilterAction,
    GroupAction
)
from atena.simulation.dataset import (
    Dataset,
    SchemaName,
    DatasetMeta,
    DatasetName,
)
from atena.simulation.display import DisplayCalculator
from atena.simulation.state import EnvStateTuple, FilteringTuple, AggregationTuple, DisplayTuple
from atena.simulation.utils import random_action_generator


class ActionExecutionResult(object):
    def __init__(self,
                 action: AbstractAction,
                 display: DisplayTuple,
                 filtered_df: pd.DataFrame,
                 aggregated_df: pd.DataFrame,
                 state: EnvStateTuple
                 ):

        self.action = action
        self.display = display
        self._filtered_df = filtered_df
        self._aggregated_df = aggregated_df
        self.state = state

    @property
    def df(self):
        if self.state.is_grouped():
            return self._aggregated_df
        return self._filtered_df


class SimulationState(object):
    def __init__(self, display_calculator: DisplayCalculator):
        empty_state = EnvStateTuple.create_empty_state()


        self.states_history = [empty_state]


        self.states_stack = [empty_state]


        disp, _ = display_calculator.calculate_display(empty_state)


        self.displays_history = [disp]


class ActionsSimulator(object):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.display_calculator = DisplayCalculator(self.dataset)
        self.simulation_state = SimulationState(self.display_calculator)

    def reset(self):

        self.display_calculator = DisplayCalculator(self.dataset)
        self.simulation_state = SimulationState(self.display_calculator)

    def run_actions(self, actions_lst: List[AbstractAction]) -> List[ActionExecutionResult]:
        self.reset()
        steps_info = []
        for action in actions_lst:
            step_info = self.execute_action(action)
            steps_info.append(step_info)

        return steps_info

    def execute_action(self, action: AbstractAction) -> ActionExecutionResult:


        if action.action_type is ActionType.BACK:

            if len(self.simulation_state.states_stack) > 1:
                self.simulation_state.states_stack.pop()
                new_state = self.simulation_state.states_stack[-1]
            else:
                new_state = EnvStateTuple.create_empty_state()

        elif action.action_type is ActionType.FILTER:
            assert isinstance(action, FilterAction)

            filt_tpl = FilteringTuple(
                column=action.filtered_column,
                term=action.filter_term,
                operator=action.filter_operator
            )
            new_state = self.simulation_state.states_stack[-1]
            new_state = new_state.append_filtering(filt_tpl)
            self.simulation_state.states_stack.append(new_state)

        elif action.action_type is ActionType.GROUP:
            assert isinstance(action, GroupAction)

            new_state = self.simulation_state.states_stack[-1]
            if action.grouped_column not in new_state["grouping"]:
                new_state = new_state.append_grouping(action.grouped_column)
            agg_tpl = AggregationTuple(
                column=action.aggregated_column,
                aggregation_function=action.aggregation_function
            )
            if agg_tpl not in new_state["aggregations"]:
                new_state = new_state.append_aggregations(agg_tpl)
            self.simulation_state.states_stack.append(new_state)
        else:
            raise Exception(f"Unknown action type: {action.action_type}")


        display, dfs = self.display_calculator.calculate_display(new_state)


        self.simulation_state.states_history.append(new_state)
        self.simulation_state.displays_history.append(display)

        return ActionExecutionResult(
            action=action,
            display=display,
            filtered_df=dfs[0],
            aggregated_df=dfs[1],
            state=new_state
        )

    def run_n_random_actions(self, n: int):
        actions_lst = [random_action_generator(self.dataset) for _ in range(n)]
        return self.run_actions(actions_lst)

    @classmethod
    def factory_from_schema_and_dataset_name(
            cls, schema_name: SchemaName,
            dataset_name: DatasetName
    ):
        dataset_meta = DatasetMeta(schema_name, dataset_name)
        dataset = Dataset(dataset_meta)
        actions_simulator = cls(dataset)

        return actions_simulator

    @classmethod
    def get_end_of_simulation_state(
            cls,
            schema_name: SchemaName,
            dataset_name: DatasetName,
            actions_lst: List[AbstractAction]
    ):
        actions_simulator = cls.factory_from_schema_and_dataset_name(schema_name, dataset_name)
        actions_simulator.run_actions(actions_lst)

        return actions_simulator.simulation_state
