import time
from atena.simulation.actions_simulator import ActionsSimulator
from revision_support.resources import query_timer

class CountedSimulator(ActionsSimulator):
    def __init__(self, dataset, owner, replay=False):
        self.owner = owner
        self.replay = replay
        super().__init__(dataset)

    def execute_action(self, action):
        preview = bool(getattr(self.owner, '_profiling_preview', False))
        category = ('preview_replay' if preview else 'replay') if self.replay else ('preview' if preview else 'commit')
        started = time.perf_counter()
        try:
            with query_timer('atena_simulator', 'execute_action', category=category, preview=preview):
                return super().execute_action(action)
        finally:
            self.owner.query_costs.counts[category] += 1
            self.owner.query_costs.seconds += time.perf_counter() - started
