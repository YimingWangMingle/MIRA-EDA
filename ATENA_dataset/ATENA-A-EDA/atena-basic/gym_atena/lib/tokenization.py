from collections import defaultdict
import re
from random import shuffle
from bisect import bisect_left, bisect_right
import math

import numpy as np
from scipy import stats

import Utilities.Configuration.config as cfg

import gym_atena.global_env_prop as gep

nan = float('nan')


def tokenize_line_of_column(line, column):
    tokens = set()


    if column != "info_line":

        tokens.add(line)

    if column == "info_line":

        for token in line.split():
            token = token.strip(",][)(")
            if "=" not in token and not token.isdigit():
                tokens.add(token)


        for token in re.findall('\[.*?\]', line):
            tokens.add(token)

    return tokens


def add_tokens_to_dict(tokens, tokens_dict, len_df):
    for token in tokens:


        if not isinstance(token, str) and math.isnan(token):
            token = nan
        tokens_dict[token] += 1 / len_df


def tokenize_column(df, column):

    tokens_dict = defaultdict(float)
    len_df = len(df)

    if len_df == 0 or column in gep.global_env_prop.env_dataset_prop.DONT_FILTER_FIELDS:
        return [], []

    for idx, line in df[column].iteritems():
        tokens = tokenize_line_of_column(line, column)
        add_tokens_to_dict(tokens, tokens_dict, len_df)


    sorted_by_freq_token_frequency_pairs = sorted(tokens_dict.items(), key=lambda kv: kv[1])
    frequencies = [val for token, val in sorted_by_freq_token_frequency_pairs]


    return sorted_by_freq_token_frequency_pairs, frequencies


def get_nearest_neighbor_token(tokens_frequency_lst, frequencies, num):

    if not tokens_frequency_lst:
        return '<UNK>'

    candidate_nns = _get_nns_candidates(tokens_frequency_lst, frequencies, num)
    if cfg.max_nn_tokens > -1:
        candidate_nns = candidate_nns[:cfg.max_nn_tokens]
    shuffle(candidate_nns)


    return candidate_nns[0][0]


def _get_nns_candidates(tokens_frequency_lst, frequencies, num):


    right_nearest_pos = bisect_left(frequencies, num)

    left_nearest_pos = bisect_right(frequencies, num) - 1

    if right_nearest_pos == len(frequencies):
        left_nearest_pos = bisect_left(frequencies, frequencies[-1])

    elif right_nearest_pos == 0:
        left_nearest_pos = 0
        right_nearest_pos = bisect_right(frequencies, frequencies[0])

    elif left_nearest_pos == right_nearest_pos:
        right_nearest_pos += 1

    elif frequencies[left_nearest_pos] - num < num - frequencies[right_nearest_pos]:
        left_nearest_pos, right_nearest_pos = right_nearest_pos, bisect_right(frequencies,
                                                                              frequencies[right_nearest_pos])

    elif frequencies[left_nearest_pos] - num > num - frequencies[right_nearest_pos]:
        right_nearest_pos, left_nearest_pos = left_nearest_pos + 1, bisect_left(frequencies,
                                                                                frequencies[right_nearest_pos - 1])

    elif frequencies[left_nearest_pos] - num == num - frequencies[right_nearest_pos]:
        right_nearest_pos = bisect_right(frequencies, frequencies[right_nearest_pos])
        left_nearest_pos = bisect_left(frequencies, frequencies[left_nearest_pos])

    else:
        raise RuntimeError("You have a logical error")


    candidate_nns = tokens_frequency_lst[left_nearest_pos: right_nearest_pos]
    return candidate_nns


def test_get_nns_candiadtes():
    frequencies = [1, 1, 2, 2, 3, 3, 4, 4]
    num = 0.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [1, 1]
    num = 1
    assert _get_nns_candidates(frequencies, frequencies, num) == [1, 1]
    num = 1.2
    assert _get_nns_candidates(frequencies, frequencies, num) == [1, 1]
    num = 1.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [1, 1, 2, 2]
    num = 1.7
    assert _get_nns_candidates(frequencies, frequencies, num) == [2, 2]
    num = 2
    assert _get_nns_candidates(frequencies, frequencies, num) == [2, 2]
    num = 2.4
    assert _get_nns_candidates(frequencies, frequencies, num) == [2, 2]
    num = 2.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [2, 2, 3, 3]
    num = 2.8
    assert _get_nns_candidates(frequencies, frequencies, num) == [3, 3]
    num = 3
    assert _get_nns_candidates(frequencies, frequencies, num) == [3, 3]
    num = 3.2
    assert _get_nns_candidates(frequencies, frequencies, num) == [3, 3]
    num = 3.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [3, 3, 4, 4]
    num = 3.7
    assert _get_nns_candidates(frequencies, frequencies, num) == [4, 4]
    num = 4
    assert _get_nns_candidates(frequencies, frequencies, num) == [4, 4]
    num = 4.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [4, 4]

    frequencies = [1]
    num = 0.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [1]
    num = 1
    assert _get_nns_candidates(frequencies, frequencies, num) == [1]
    num = 1.5
    assert _get_nns_candidates(frequencies, frequencies, num) == [1]


if __name__ == '__main__':
    test_get_nns_candiadtes()
