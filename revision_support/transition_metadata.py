import numpy as np
from .reference_policy import state_identity


def dual_reference_metadata(env):
    if hasattr(env, 'candidate_action_ids'):
        masks = np.zeros((env.set_action_dim, env.operation_action_dim), dtype=bool)
        for slot, actions in enumerate(env.candidate_action_ids):
            masks[slot, actions] = True
        if not masks.any():


            masks[:, np.asarray(env.base_env.valid_actions(), dtype=np.int64)] = True
        context = (int(env.base_env.current_set_id), tuple(env.candidate_set_ids),
                   int(env.step_count), sorted(env.base_env.found_items_with_ratio.items()))
    else:
        masks = np.zeros((env.set_action_space.n, env.operation_action_space.n), dtype=bool)
        datasets = env.datasets or [env.input_set]
        for slot, dataset in enumerate(datasets):
            masks[slot] = np.asarray(env.action_manager.fix_possible_operation_action_probs(
                dataset, [1.] * env.operation_action_space.n)) > 0

        definitions = tuple((tuple((p.attribute, p.operator, p.value, p.is_category,
            p.is_multiple_line_string) for p in d.predicate.components),
            d.predicate.logical_operator, d.set_id,
            tuple(tuple(sorted(vars(j).items())) for j in d.joins), tuple(d.tables)) for d in datasets)
        context = (definitions, int(env.step_count),
                   np.asarray(getattr(env, 'example_state', [])).tolist(),
                   sorted(env.state_encoder.found_items_with_ratio.items(), key=lambda item: str(item[0])))
    return masks.reshape(-1), state_identity(context)


def atena_reference_identity(env):
    state = env.simulator.simulation_state
    return state_identity(env.schema_name, env.dataset_name, env.step_index,
                          tuple(state.states_stack), tuple(state.states_history),
                          tuple(tuple(sorted(vars(action).items())) for action in env.actions))
