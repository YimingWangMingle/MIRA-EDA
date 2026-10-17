def score_returned_sets(encoder, datasets, parent):
    encoded, reward, interestingness = encoder.encode_datasets(datasets, parent)
    width = len(encoder.set_description)
    states = [encoded[i * width:(i + 1) * width] for i in range(len(datasets))]
    return states, float(reward), float(interestingness)


def log_returned_sets(datasets, states, episode_ids, logged_ids, state_writer,
                      trace_writer, prefix, suffix, reward, interestingness):
    count = 0
    for dataset, state in zip(datasets, states):
        try:
            set_id = int(dataset.set_id)
        except (TypeError, ValueError):
            continue
        if set_id < 0:
            continue
        episode_ids.add(set_id)
        if set_id not in logged_ids:
            logged_ids.add(set_id)
            state_writer.writerow([set_id] + list(state))
        trace_writer.writerow(prefix + [set_id, reward, interestingness] + suffix)
        count += 1
    if not count:
        trace_writer.writerow(prefix + [-1, reward, interestingness] + suffix)
