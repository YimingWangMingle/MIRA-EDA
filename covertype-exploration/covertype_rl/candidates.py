"""Shared target-blind candidate pool for Covertype policies."""
import numpy as np

OPERATOR_FAMILIES = ("by_facet", "by_superset", "by_neighbors", "by_distribution")

def _candidate_groups(universe, actions, set_id):
    set_id = int(set_id)
    row = np.asarray(universe.graph[set_id], dtype=np.int64)
    grouped = {}
    family_by_set = {}
    for action_id, next_set_id in enumerate(row.tolist()):
        next_set_id = int(next_set_id)
        if next_set_id == set_id:
            continue
        grouped.setdefault(next_set_id, []).append(int(action_id))
        family_by_set.setdefault(next_set_id, set()).add(
            str(actions[action_id].family)
        )
    return list(grouped), grouped, family_by_set

def _select_candidate_set_ids(
    universe,
    all_candidate_set_ids,
    family_by_set,
    reference_set_size,
    candidate_slots,
):
    if len(all_candidate_set_ids) <= int(candidate_slots):
        return [int(set_id) for set_id in all_candidate_set_ids]

    selected = []
    selected_set = set()

    def add(set_id):
        set_id = int(set_id)
        if set_id in selected_set or len(selected) >= int(candidate_slots):
            return False
        selected.append(set_id)
        selected_set.add(set_id)
        return True

    for family in OPERATOR_FAMILIES:
        family_candidates = [
            int(set_id)
            for set_id in all_candidate_set_ids
            if family in family_by_set.get(int(set_id), set())
        ]
        if not family_candidates:
            continue
        family_candidates.sort(
            key=lambda set_id: (
                abs(
                    np.log1p(float(universe.size_for_set(set_id)))
                    - np.log1p(float(reference_set_size))
                ),
                int(set_id),
            )
        )
        add(family_candidates[0])

    remaining = [
        int(set_id)
        for set_id in all_candidate_set_ids
        if int(set_id) not in selected_set
    ]
    remaining.sort(
        key=lambda set_id: (float(universe.size_for_set(set_id)), int(set_id))
    )
    if remaining:
        quantile_indices = np.linspace(
            0,
            len(remaining) - 1,
            max(1, int(candidate_slots) - len(selected)),
        )
        for index in quantile_indices.round().astype(int).tolist():
            add(remaining[int(index)])

    if len(selected) < int(candidate_slots):
        selected_states = [universe.state_for_set(set_id) for set_id in selected]
        leftovers = [set_id for set_id in remaining if set_id not in selected_set]
        while leftovers and len(selected) < int(candidate_slots):
            if not selected_states:
                add(leftovers.pop(0))
                selected_states.append(universe.state_for_set(selected[-1]))
                continue
            best_position = 0
            best_distance = -1.0
            for position, set_id in enumerate(leftovers):
                state = universe.state_for_set(set_id)
                distance = min(
                    float(np.linalg.norm(state - existing))
                    for existing in selected_states
                )
                if distance > best_distance:
                    best_distance = distance
                    best_position = position
            chosen = leftovers.pop(best_position)
            if add(chosen):
                selected_states.append(universe.state_for_set(chosen))
    return selected

def build_candidate_view(universe, actions, current_set_id, candidate_slots=10):
    all_set_ids, grouped, family_by_set = _candidate_groups(
        universe,
        actions,
        current_set_id,
    )
    selected = _select_candidate_set_ids(
        universe,
        all_set_ids,
        family_by_set,
        reference_set_size=universe.size_for_set(current_set_id),
        candidate_slots=candidate_slots,
    )
    action_ids = [list(grouped[int(set_id)]) for set_id in selected]
    while len(selected) < int(candidate_slots):
        selected.append(-1)
        action_ids.append([])
    return selected[: int(candidate_slots)], action_ids[: int(candidate_slots)]
