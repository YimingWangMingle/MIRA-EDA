from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    state: np.ndarray
    interestingness: float


@dataclass(frozen=True)
class GreedySelection:
    index: int
    candidate_id: int
    score: float
    components: dict


@dataclass(frozen=True)
class CandidateScoreBatch:


    base_scores: np.ndarray
    interestingness: np.ndarray
    coherency: np.ndarray
    diversity: np.ndarray


def _finite_vector(value):
    return np.nan_to_num(
        np.asarray(value, dtype=np.float64).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _unit_vector(value):
    vector = _finite_vector(value)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def _minmax(values):
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def score_candidates(
    candidates,
    current_state,
    history_states,
    eta=0.1,
):
    if not candidates:
        raise ValueError("Greedy EDA has no legal candidates")

    current = _unit_vector(current_state)
    states = [_unit_vector(candidate.state) for candidate in candidates]
    history = [_unit_vector(state) for state in history_states]

    interestingness = _minmax(
        [candidate.interestingness for candidate in candidates]
    )
    coherency = _minmax(
        [(float(np.dot(current, state)) + 1.0) / 2.0 for state in states]
    )
    diversity = _minmax(
        [
            1.0
            if not history
            else 1.0
            - np.exp(
                -float(eta)
                * min(float(np.linalg.norm(state - old)) for old in history)
            )
            for state in states
        ]
    )
    base_scores = interestingness + coherency + diversity
    return CandidateScoreBatch(
        base_scores=base_scores,
        interestingness=interestingness,
        coherency=coherency,
        diversity=diversity,
    )


def choose_candidate(
    candidates,
    current_state,
    history_states,
    visited_ids,
    rng,
    eta=0.1,
    repeat_penalty=1.0,
):
    batch = score_candidates(candidates, current_state, history_states, eta=eta)
    repeats = np.asarray(
        [
            float(repeat_penalty)
            if int(candidate.candidate_id) in visited_ids
            else 0.0
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    scores = batch.base_scores - repeats
    best = np.flatnonzero(
        np.isclose(scores, float(scores.max()), rtol=0.0, atol=1e-12)
    )
    index = int(rng.choice(best))

    return GreedySelection(
        index=index,
        candidate_id=int(candidates[index].candidate_id),
        score=float(scores[index]),
        components={
            "interestingness": float(batch.interestingness[index]),
            "coherency": float(batch.coherency[index]),
            "diversity": float(batch.diversity[index]),
            "repeat_penalty": float(repeats[index]),
        },
    )


def choose_operation(
    legal_action_ids,
    family_by_action,
    family_counts,
    action_counts,
    rng,
):
    legal = [int(action_id) for action_id in legal_action_ids]
    if not legal:
        raise ValueError("Greedy EDA has no legal operations")

    minimum_family_count = min(
        family_counts.get(family_by_action[action_id], 0) for action_id in legal
    )
    least_used_families = {
        family_by_action[action_id]
        for action_id in legal
        if family_counts.get(family_by_action[action_id], 0)
        == minimum_family_count
    }
    family_legal = [
        action_id
        for action_id in legal
        if family_by_action[action_id] in least_used_families
    ]
    minimum_action_count = min(
        action_counts.get(action_id, 0) for action_id in family_legal
    )
    tied = [
        action_id
        for action_id in family_legal
        if action_counts.get(action_id, 0) == minimum_action_count
    ]
    return int(rng.choice(np.asarray(tied, dtype=np.int64)))
