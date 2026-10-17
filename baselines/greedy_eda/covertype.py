from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .policy import GreedySelection, score_candidates


@dataclass
class WorkerExplorationMemory:


    set_visit_counts: Dict[int, int] = field(default_factory=dict)

    def visit_count(self, set_id):
        return int(self.set_visit_counts.get(int(set_id), 0))

    def counts_for(self, candidates):
        return np.asarray(
            [self.visit_count(candidate.candidate_id) for candidate in candidates],
            dtype=np.int64,
        )

    def register_root(self, set_id):
        set_id = int(set_id)
        if set_id not in self.set_visit_counts:
            self.set_visit_counts[set_id] = 1

    def record_visit(self, set_id, valid_transition=True):
        if not valid_transition:
            return
        set_id = int(set_id)
        self.set_visit_counts[set_id] = self.visit_count(set_id) + 1


def choose_count_balanced_candidate(
    candidates,
    current_state,
    history_states,
    memory,
    rng,
    eta=0.1,
    top_k=3,
):

    if int(top_k) < 1:
        raise ValueError("top_k must be at least 1")

    batch = score_candidates(candidates, current_state, history_states, eta=eta)
    visit_counts = memory.counts_for(candidates)
    eligible = np.flatnonzero(visit_counts == visit_counts.min()).tolist()
    eligible.sort(
        key=lambda index: (
            -float(batch.base_scores[index]),
            int(candidates[index].candidate_id),
        )
    )
    ranked = eligible[: min(int(top_k), len(eligible))]

    if len(ranked) == 1:
        index = ranked[0]
    else:
        weights = np.arange(len(ranked), 0, -1, dtype=np.float64)
        probabilities = weights / weights.sum()
        rank = int(rng.choice(len(ranked), p=probabilities))
        index = ranked[rank]

    return GreedySelection(
        index=index,
        candidate_id=int(candidates[index].candidate_id),
        score=float(batch.base_scores[index]),
        components={
            "interestingness": float(batch.interestingness[index]),
            "coherency": float(batch.coherency[index]),
            "diversity": float(batch.diversity[index]),
            "worker_visit_count": int(visit_counts[index]),
        },
    )
