import math
import random
from collections import Counter

import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

import networkx as nx


from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
from nltk.translate import bleu_score as nltk_bleu
from zss import Node, simple_distance

from gym_atena.envs.atena_env_cont import ATENAEnvCont
from gym_atena.lib.helpers import (
    OPERATOR_TYPE_LOOKUP,
    INT_OPERATOR_MAP_ATENA_STR,
)
import gym_atena.global_env_prop as gep
from gym_atena.reactida.utils.distance import display_distance


class PositiveNegativeStats(object):

    def __init__(self, TP, FP, FN):
        self.TP = TP
        self.FP = FP
        self.FN = FN

    @property
    def precision(self):
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self):
        return self.TP / (self.TP + self.FN)

    @property
    def f1(self):
        if (self.precision + self.recall) == 0:
            return 0

        return 2 * self.precision * self.recall / (self.precision + self.recall)


def precision_score_without_back(references, candidate, back_token):

    positive_negative_stats = positive_negative_stats_without_back(references, candidate, back_token=back_token)
    return positive_negative_stats.precision


def positive_negative_stats_without_back(references, candidate, back_token):

    references = remove_back_tokens_from_nested_lists(references, back_token=back_token)
    candidate = remove_back_tokens_from_nested_lists(candidate, back_token=back_token)

    candidate = list(set(candidate))
    candidate_success = [False] * len(candidate)
    for idx, cand_token in enumerate(candidate):
        cand_token_found = False
        for reference in references:
            for ref_token in reference:
                if ref_token == cand_token:
                    candidate_success[idx] = True
                    cand_token_found = True
                    break
            if cand_token_found:
                break

    references_tokens = list(set([ref_token for reference in references for ref_token in reference]))
    references_success = [False] * len(references_tokens)
    for idx, ref_token in enumerate(references_tokens):
        for cand_token in candidate:
            if ref_token == cand_token:
                references_success[idx] = True
                break

    TP, FP = candidate_success.count(True), candidate_success.count(False)
    FN = references_success.count(False)

    return PositiveNegativeStats(TP, FP, FN)


def micro_precision_without_back(references, candidates, back_token):

    true_positives = false_positives = 0
    for candidate in candidates:
        positive_negative_stats = positive_negative_stats_without_back(references, candidate, back_token)
        true_positives += positive_negative_stats.TP
        false_positives += positive_negative_stats.FP
    return true_positives / (true_positives + false_positives)


def micro_recall_without_back(references, candidates, back_token):

    true_positives = false_negatives = 0
    for candidate in candidates:
        positive_negative_stats = positive_negative_stats_without_back(references, candidate, back_token)
        true_positives += positive_negative_stats.TP
        false_negatives += positive_negative_stats.FN
    return true_positives / (true_positives + false_negatives)


def micro_f1_without_back(references, candidates, back_token):

    precision = micro_precision_without_back(references, candidates, back_token)
    recall = micro_recall_without_back(references, candidates, back_token)
    if (precision + recall) == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def recall_score_without_back(references, candidate, back_token):

    positive_negative_stats = positive_negative_stats_without_back(references, candidate, back_token=back_token)
    return positive_negative_stats.recall


def f1_score_without_back(references, candidate, back_token):

    precision = precision_score_without_back(references, candidate, back_token=back_token)
    recall = recall_score_without_back(references, candidate, back_token=back_token)

    if (precision + recall) == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def paired_pvalue(lst1, lst2):

    assert len(lst1) == len(lst2)
    a = np.hstack(
        np.array(
            [np.array(lst1[i])-np.array(lst2[i]) for i in range(len(lst1))]
        )
    )
    b = np.zeros_like(a)
    t_statistic, p_val = sp.stats.ttest_ind(a, b)

    p_val /= 2
    return t_statistic, p_val


def remove_back_tokens_from_nested_lists(lst, back_token):

    if isinstance(lst, list):
        new_lst = []
        for elem in lst:
            new_elem = remove_back_tokens_from_nested_lists(elem, back_token)
            if new_elem is not None:
                new_lst.append(new_elem)
        return new_lst
    else:
        if lst == back_token:
            return None
        return lst


def corpus_bleu_without_back(
        list_of_references,
        hypotheses,
        back_token,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
):

    list_of_references = remove_back_tokens_from_nested_lists(list_of_references, back_token)
    hypotheses = remove_back_tokens_from_nested_lists(hypotheses, back_token)
    return corpus_bleu(list_of_references,
                       hypotheses,
                       weights=weights,
                       smoothing_function=smoothing_function,
                       auto_reweigh=auto_reweigh
                       )


def sentence_bleu_without_back(
        references,
        hypothesis,
        back_token,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
):

    return corpus_bleu_without_back(
        [references],
        [hypothesis],
        back_token=back_token,
        weights=weights,
        smoothing_function=smoothing_function,
        auto_reweigh=auto_reweigh)


def every_tree_paths(tree, min_len, max_len, back_token='[back]'):


    if max_len == -1:
        max_len = len(tree)
    for n in range(min_len, max_len + 1):
        for ng in tree_paths(tree, n, back_token=back_token):
            yield ng


def tree_paths(
        tree,
        n,
        back_token='[back]',
):

    sequence = iter(tree)
    history = []
    while sequence:

        try:
            next_item = next(sequence)
        except StopIteration:

            return
        if next_item == back_token:
            if history:
                history.pop()
        else:
            history.append(next_item)
            if len(history) >= n:
                yield (tuple(history[-n:]))


def tree_corpus_bleu(
        list_of_references,
        hypotheses,
        back_token,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
):


    p_numerators = Counter()
    p_denominators = Counter()
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), (
        "The number of hypotheses and their reference(s) should be the " "same "
    )


    for references, hypothesis in zip(list_of_references, hypotheses):


        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i, back_token=back_token)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator


        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += nltk_bleu.closest_ref_length(references, hyp_len)


    bp = nltk_bleu.brevity_penalty(ref_lengths, hyp_lengths)


    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths


    p_n = [
        nltk_bleu.Fraction(p_numerators[i], p_denominators[i], _normalize=False)
        for i, _ in enumerate(weights, start=1)
    ]


    if p_numerators[1] == 0:
        return 0


    if not smoothing_function:
        smoothing_function = nltk_bleu.SmoothingFunction().method0


    p_n = smoothing_function(
        p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths
    )
    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    s = bp * math.exp(math.fsum(s))
    return s


def tree_corpus_gleu(list_of_references, hypotheses, back_token, min_len=1, max_len=4):


    assert len(list_of_references) == len(
        hypotheses
    ), "The number of hypotheses and their reference(s) should be the same"


    corpus_n_match = 0
    corpus_n_all = 0

    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_ngrams = Counter(every_tree_paths(hypothesis, min_len, max_len, back_token=back_token))
        tpfp = sum(hyp_ngrams.values())

        hyp_counts = []
        for reference in references:
            ref_ngrams = Counter(every_tree_paths(reference, min_len, max_len, back_token=back_token))
            tpfn = sum(ref_ngrams.values())

            overlap_ngrams = ref_ngrams & hyp_ngrams
            tp = sum(overlap_ngrams.values())


            n_all = max(tpfp, tpfn)

            if n_all > 0:
                hyp_counts.append((tp, n_all))


        if hyp_counts:
            n_match, n_all = max(hyp_counts, key=lambda hc: hc[0] / hc[1])
            corpus_n_match += n_match
            corpus_n_all += n_all


    if corpus_n_all == 0:
        gleu_score = 0.0
    else:
        gleu_score = corpus_n_match / corpus_n_all

    return gleu_score

def tree_sentence_gleu(references, hypothesis, min_len=1, max_len=4, back_token='[back]'):

    return tree_corpus_gleu([references], [hypothesis], min_len=min_len, max_len=max_len, back_token=back_token)


def tree_sentence_gleu_n(list_of_references, hypotheses, back_token, n):

    return tree_sentence_gleu(list_of_references, hypotheses, back_token=back_token, min_len=1, max_len=n)


def tree_corpus_gleu_n(list_of_references, hypotheses, back_token, n):

    return tree_corpus_gleu(list_of_references, hypotheses, back_token=back_token, min_len=1, max_len=n)

def tree_corpus_bleu_n(
        list_of_references,
        hypotheses,
        back_token,
        n,
        smoothing_function=None,
        auto_reweigh=False,
):

    if n == 1:
        weights = (1,)
    elif n == 2:
        weights = (1/2, 1/2)
    elif n ==3:
        weights = (1/3, 1/3, 1/3)
    elif n == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        weights = (1/n for i in range(n))

    return tree_corpus_bleu(list_of_references, hypotheses, back_token=back_token, weights=weights,
                            smoothing_function=smoothing_function, auto_reweigh=auto_reweigh)


def modified_precision(references, hypothesis, n, back_token):


    counts = Counter(tree_paths(hypothesis, n, back_token=back_token)) if len(hypothesis) >= n else Counter()


    max_counts = {}
    for reference in references:
        reference_counts = (
            Counter(tree_paths(reference, n, back_token=back_token)) if len(reference) >= n else Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])


    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = sum(clipped_counts.values())


    denominator = max(1, sum(counts.values()))

    return nltk_bleu.Fraction(numerator, denominator, _normalize=False)


def tree_sentence_bleu(
        references,
        hypothesis,
        back_token,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
):

    return tree_corpus_bleu(
        [references], [hypothesis], back_token, weights, smoothing_function, auto_reweigh
    )


def tree_sentence_bleu_n(
        references,
        hypothesis,
        back_token,
        n,
        smoothing_function=None,
        auto_reweigh=False,
):

    return tree_corpus_bleu_n(
        [references], [hypothesis], back_token, n, smoothing_function, auto_reweigh
    )


def get_refs_and_multiple_cands_tokens_from_actions(all_datasets_refs_acts, all_datasets_cands_acts,
                                           compressed_ref=False,
                                           filter_by_field_ref=True,
                                           continuous_filter_term_ref=True,
                                           compressed_cand=False,
                                           filter_by_field_cand=True,
                                           continuous_filter_term_cand=True
                                           ):

    token = 0
    displays_token_map = dict()
    back_token = '[back]'

    num_of_datasets = len(all_datasets_cands_acts[0])
    references = [[] for _ in range(num_of_datasets)]
    candidates = [[] for _ in range(num_of_datasets)]


    for cand_num, all_datasets_cand_acts in enumerate(all_datasets_cands_acts):
        for dataset_number, (dataset_refs_acts, dataset_cand_acts) in enumerate(zip(all_datasets_refs_acts, all_datasets_cand_acts)):
            to_convert = dataset_refs_acts + [dataset_cand_acts]
            for idx, act_lst in enumerate(to_convert):
                if idx == len(to_convert)-1:
                    dhist, ahist, info_hist = ATENAEnvCont.get_sessions_hists(
                        act_lst,
                        dataset_number=dataset_number,
                        compressed=compressed_cand,
                        filter_by_field=filter_by_field_cand,
                        continuous_filter_term=continuous_filter_term_cand
                    )
                else:
                    dhist, ahist, info_hist = ATENAEnvCont.get_sessions_hists(
                        act_lst,
                        dataset_number=dataset_number,
                        compressed=compressed_ref,
                        filter_by_field=filter_by_field_ref,
                        continuous_filter_term=continuous_filter_term_ref
                    )


                ahist = [None] + ahist

                tokens = []
                for disp, act in zip(dhist, ahist):
                    disp = str(disp)

                    if disp not in displays_token_map:
                        displays_token_map[disp] = token
                        token += 1


                    act_str = None if act is None else OPERATOR_TYPE_LOOKUP[act[0]]
                    if act_str == "back":
                        tokens.append(back_token)
                    else:
                        tokens.append(displays_token_map[disp])

                if idx == len(to_convert)-1:
                    candidates[dataset_number].append(tokens)
                else:
                    references[dataset_number].append(tokens)

    return references, candidates


def get_refs_and_cands_tokens_from_actions(all_datasets_refs_acts, all_datasets_cand_acts,
                                           compressed_ref=False,
                                           filter_by_field_ref=True,
                                           continuous_filter_term_ref=True,
                                           compressed_cand=False,
                                           filter_by_field_cand=True,
                                           continuous_filter_term_cand=True
                                           ):

    token = 0
    displays_token_map = dict()
    back_token = '[back]'

    references = []
    candidates = []


    for dataset_number, (dataset_refs_acts, dataset_cand_acts) in enumerate(zip(all_datasets_refs_acts, all_datasets_cand_acts)):
        to_convert = dataset_refs_acts + [dataset_cand_acts]
        dataset_references = []
        dataset_candidate = None
        for idx, act_lst in enumerate(to_convert):
            if idx == len(to_convert)-1:
                dhist, ahist, info_hist = ATENAEnvCont.get_sessions_hists(
                    act_lst,
                    dataset_number=dataset_number,
                    compressed=compressed_cand,
                filter_by_field=filter_by_field_cand,
                    continuous_filter_term=continuous_filter_term_cand
                )
            else:
                dhist, ahist, info_hist = ATENAEnvCont.get_sessions_hists(
                    act_lst,
                    dataset_number=dataset_number,
                    compressed=compressed_ref,
                    filter_by_field=filter_by_field_ref,
                    continuous_filter_term=continuous_filter_term_ref
                )


            ahist = [None] + ahist

            tokens = []
            for disp, act in zip(dhist, ahist):
                disp = str(disp)

                if disp not in displays_token_map:
                    displays_token_map[disp] = token
                    token += 1


                act_str = None if act is None else OPERATOR_TYPE_LOOKUP[act[0]]
                if act_str == "back":
                    tokens.append(back_token)
                else:
                    tokens.append(displays_token_map[disp])

            if idx == len(to_convert)-1:
                dataset_candidate = tokens
            else:
                dataset_references.append(tokens)

        references.append(dataset_references)
        candidates.append(dataset_candidate)

    return references, candidates


def construct_displays_tree(dhist, ahist):
    tree_size = 1
    root = Node(dhist[0])
    stack = [root]

    nodes_left = dhist[1:]

    for disp, act in zip(nodes_left, ahist):
        act_string = OPERATOR_TYPE_LOOKUP[act[0]]
        if act_string == "back":
            if len(stack) > 1:
                stack.pop()
        else:
            tree_size += 1
            new_node = Node(disp)
            stack[-1].addkid(new_node)
            stack.append(new_node)

    return root, tree_size


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):


    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):


        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def construct_nx_displays_tree(dhist, ahist, info_hist):
    G = nx.DiGraph()
    edge_labels = {}
    stack = [0]

    nodes_left = dhist[1:]

    for node_id, (disp, act, info) in enumerate(zip(nodes_left, ahist, info_hist), start=1):
        act_string = OPERATOR_TYPE_LOOKUP[act[0]]
        act_column = gep.global_env_prop.env_dataset_prop.KEYS_ANALYST_STR[act[1]]
        if act_string == "back":
            if len(stack) > 1:
                stack.pop()
        else:
            G.add_edge(stack[-1], node_id, act=info["action"])

            filter_operator = "" if act_string == "group" else ", " + INT_OPERATOR_MAP_ATENA_STR[act[2]]
            filter_term = "" if act_string == "group" else "\n" + str(info["filter_term"])
            edge_labels[(stack[-1], node_id)] = f'{act_string[0].upper()}, {act_column}{filter_operator}{filter_term}'
            stack.append(node_id)

    return G, edge_labels


def construct_nx_display_tree_from_actions_lst(actions_lst, dataset_number, filter_terms_lst=None):
    dhist, ahist, info_hist = ATENAEnvCont.get_sessions_hists(
        actions_lst, dataset_number=dataset_number, filter_terms_lst=filter_terms_lst)


    G, edge_labels = construct_nx_displays_tree(dhist, ahist, info_hist)

    return G, edge_labels


def get_number_of_back_actions_in_the_end(actions_lst):
    count = 0
    reverse_actions_lst = actions_lst[::-1]

    while reverse_actions_lst[count][0] == 0:
        count += 1

    return count


def draw_nx_display_tree(actions_lst, dataset_number, filter_terms_lst=None, plot_path=None, show=True, hists=None):

    if hists:
        info_hist, dhist, ahist = hists
        G, edge_labels = construct_nx_displays_tree(dhist, ahist, info_hist)
    else:
        G, edge_labels = construct_nx_display_tree_from_actions_lst(actions_lst, dataset_number, filter_terms_lst=
        filter_terms_lst)


    pos = hierarchy_pos(G, 0)

    figw, figh = 20.0, 8.0
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(figw, figh))


    num_of_back_in_the_end = get_number_of_back_actions_in_the_end(actions_lst)
    last_node = list(G.nodes())[-1]
    while num_of_back_in_the_end and list(G.in_edges(last_node))[-1]:
        num_of_back_in_the_end -= 1
        last_node = list(G.in_edges(last_node))[-1][0]
    current_node_in_tree = last_node
    node_colors = ['green' if node == current_node_in_tree else 'red'for node in G.nodes]

    nx.draw_networkx(G, pos=pos, node_color=node_colors)
    text = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    for _, t in text.items():
        t.set_rotation('horizontal')
    if plot_path:
        plt.savefig(plot_path)

    if show:
        plt.show()
    else:
        plt.close()

    return G


def draw_ref_and_candidate_trees_side_by_side(ref, cand, dataset_number):
    region = 120
    figw, figh = 18.0, 16.0
    fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(figw, figh))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.01, hspace=0.01)
    for actions_lst, title in [(ref, "reference"), (cand, "candidate")]:
        G, edge_labels = construct_nx_display_tree_from_actions_lst(actions_lst, dataset_number)
        pos = hierarchy_pos(G, 0)
        region += 1
        plt.subplot(region)
        plt.title(title)
        nx.draw_networkx(G, pos=pos)
        text = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
        for _, t in text.items():
            t.set_rotation('horizontal')

    plt.show()


def display_label_sitance(disp1, disp2, normalize, unit=5000):
    if not normalize:
        unit = 1

    if disp1 == disp2 == "":
        return 0
    elif disp1 == "" or disp2 == "":
        return unit


    return display_distance(disp1, disp2).display_distance * unit * 2.0


def compute_display_TED(root1, root2, normalize):
    cost, ops =  simple_distance(
        root1, root2, label_dist=lambda x, y: display_label_sitance(x, y, normalize), return_operations=True)
    return cost, ops


def compute_minimum_display_TED_from_actions(references,
                                             candidate,
                                             dataset_number,
                                             normalize=True,
                                             compressed_ref=False,
                                             filter_by_field_ref=True,
                                             continuous_filter_term_ref=True,
                                             compressed_cand=False,
                                             filter_by_field_cand=True,
                                             continuous_filter_term_cand=True,
                                             return_min_ops=False
                                             ):
    references_dhist_ahist_pairs = [
        ATENAEnvCont.get_sessions_hists(
            act_lst, dataset_number=dataset_number,
            compressed=compressed_ref,
            filter_by_field=filter_by_field_ref,
            continuous_filter_term=continuous_filter_term_ref
        ) for act_lst
        in references
    ]

    cand_dhist, cand_ahist, info_hist = ATENAEnvCont.get_sessions_hists(
        candidate, dataset_number=dataset_number,
        compressed=compressed_cand,
        filter_by_field=filter_by_field_cand,
        continuous_filter_term=continuous_filter_term_cand
    )
    cand_tree, cand_tree_size = construct_displays_tree(cand_dhist, cand_ahist)


    teds_lst = []
    ops_lst = []

    for dhist, ahist, info_hist in references_dhist_ahist_pairs:
        ref_tree, ref_tree_size = construct_displays_tree(dhist, ahist)
        ted, ops = compute_display_TED(cand_tree, ref_tree, normalize)
        if normalize:
            ted = 2 * ted / ((cand_tree_size + ref_tree_size) * 5000 + ted)
        teds_lst.append(ted)
        ops_lst.append(ops)


    if return_min_ops:
        min_ops = ops_lst[np.argmin(teds_lst)]
        return min(teds_lst), np.argmin(teds_lst), teds_lst, min_ops
    return min(teds_lst), np.argmin(teds_lst), teds_lst


if __name__ == '__main__':


    actions_lst = [
        [2, 6, 2, 0.3221167325973511, 0, 0],
        [0, 3, 3, -0.45003635857105255, 0, 0],
        [2, 6, 2, 0.3221167325973511, 0, 0],


    ]

    actions_lst2 = [
        [2, 2, 0, 0.0, 0, 0],
        [0, 4, 8, -0.4284098446369171, 0, 0],
        [2, 2, 0, 0.0, 0, 0],
        [1, 6, 2, -0.17132456600666046, 0, 0],


    ]

    references = [[[2, 3, 1, 0.3221167325973511, 0, 0],
                   [0, 5, 4, 0.6, 0, 0],
                   [2, 2, 8, 0.21705351769924164, 0, 0],
                   [2, 6, 8, -0.49803635857105255, 0, 0],
                   [1, 4, 7, -0.46546758365631, 0, 0],
                   [0, 10, 8, 0.38086163997650146, 0, 0],
                   [0, 3, 1, -0.4, 0, 0],
                   [0, 6, 8, 0.444186806678772, 0, 0],
                   [1, 3, 4, 0.6415030360221863, 0, 0],
                   [2, 10, 4, 0.5431699156761169, 0, 0],
                   [2, 6, 1, 0.5681825876235962, 0, 0],
                   [1, 6, 4, 0.8115951418876648, 0, 0]],
                  [[2, 3, 1, 0.3221167325973511, 0, 0],
                   [1, 3, 1, 0.6, 0, 0],
                   [2, 6, 8, 0.21705351769924164, 0, 0],
                   [0, 6, 8, -0.49803635857105255, 0, 0],
                   [0, 4, 7, -0.46546758365631, 0, 0],
                   [0, 10, 8, 0.38086163997650146, 0, 0],
                   [1, 4, 7, -0.44, 0, 0],
                   [2, 5, 8, 0.444186806678772, 0, 0],
                   [2, 9, 4, 0.6415030360221863, 0, 0],
                   [0, 5, 1, 0.5431699156761169, 0, 0],
                   [0, 5, 1, 0.5681825876235962, 0, 0],
                   [1, 5, 1, 0.5431699156761169, 0, 0]],
                  [[2, 2, 1, 0.3221167325973511, 0, 0],
                   [2, 6, 1, 0.6, 0, 0],
                   [0, 6, 8, 0.21705351769924164, 0, 0],
                   [0, 6, 8, -0.49803635857105255, 0, 0],
                   [2, 3, 7, -0.46546758365631, 0, 0],
                   [2, 5, 8, 0.38086163997650146, 0, 0],
                   [0, 4, 7, -0.44, 0, 0],
                   [0, 5, 8, 0.444186806678772, 0, 0],
                   [1, 6, 4, 0.5415030360221863, 0, 0],
                   [2, 6, 1, 0.5431699156761169, 0, 0],
                   [2, 11, 1, -0.5681825876235962, 0, 0],
                   [2, 10, 1, 0.5431699156761169, 0, 0]],
                  [[2, 6, 1, 0.3221167325973511, 0, 0],
                   [2, 5, 1, 0.6, 0, 0],
                   [0, 6, 8, 0.21705351769924164, 0, 0],
                   [0, 6, 8, -0.49803635857105255, 0, 0],
                   [2, 3, 7, -0.46546758365631, 0, 0],
                   [1, 3, 1, -0.08086163997650146, 0, 0],
                   [2, 9, 7, -0.44, 0, 0],
                   [2, 5, 8, 0.444186806678772, 0, 0],
                   [1, 5, 7, -0.4815030360221863, 0, 0],
                   [2, 10, 1, 0.5431699156761169, 0, 0],
                   [0, 10, 1, -0.4801825876235962, 0, 0],
                   [2, 6, 1, 0.5431699156761169, 0, 0]],
                  [[2, 2, 1, 0.3221167325973511, 0, 0],
                   [2, 3, 1, 0.6, 0, 0],
                   [2, 6, 8, 0.21705351769924164, 0, 0],
                   [0, 6, 8, -0.49803635857105255, 0, 0],
                   [0, 2, 7, -0.46546758365631, 0, 0],
                   [2, 11, 1, -0.08086163997650146, 0, 0],
                   [0, 3, 7, -0.44, 0, 0],
                   [0, 5, 8, 0.444186806678772, 0, 0],
                   [1, 6, 1, 0.4815030360221863, 0, 0],
                   [2, 5, 1, 0.5431699156761169, 0, 0],
                   [1, 5, 1, -0.00501825876235962, 0, 0],
                   [2, 9, 1, 0.5431699156761169, 0, 0]],
                  [[2, 6, 0, 0.0, 0, 0],
                   [2, 5, 1, 0.03472083806991577, 0, 0],
                   [0, 0, 0, 0.0, 0, 0],
                   [0, 0, 0, 0.0, 0, 0],
                   [1, 5, 1, -0.05709300562739372, 0, 0],
                   [1, 3, 1, -0.3, 0, 0],
                   [2, 6, 0, 0.0, 0, 0],
                   [0, 0, 0, 0.0, 0, 0],
                   [0, 0, 0, 0.0, 0, 0],
                   [1, 3, 2, 0.12554802000522614, 0, 0],
                   [2, 10, 0, 0.0, 0, 0],
                   [2, 6, 0, 0.0, 0, 0]],
                  [[2, 2, 0, 0.0, 0, 0],
                  [2, 3, 1, 0.03472083806991577, 0, 0],
                  [2, 6, 0, 0.0, 0, 0],
                  [1, 4, 7, -0.46546758365631, 0, 0],
                  [0, 5, 1, -0.05709300562739372, 0, 0],
                  [0, 3, 1, -0.3, 0, 0],
                  [0, 6, 0, 0.0, 0, 0],
                  [0, 0, 0, 0.0, 0, 0],
                  [1, 2, 1, -0.3, 0, 0],
                  [2, 3, 2, 0.12554802000522614, 0, 0],
                  [2, 10, 0, 0.0, 0, 0],
                  [2, 6, 0, 0.0, 0, 0]],
                  ]
    candidate = [[2, 2, 0, 0.5, 0, 0],
                 [2, 3, 0, 0.5, 0, 0],
                 [2, 5, 0, 0.5, 0, 0],
                 [1, 2, 8, 0.21678057312965393-0.5, 0, 0],
                 [0, 0, 0, 0.5, 0, 0],
                 [0, 0, 0, 0.5, 0, 0],
                 [0, 0, 0, 0.5, 0, 0],
                 [0, 0, 0, 0.5, 0, 0],
                 [1, 5, 8, 0.17377719283103943-0.5, 0, 0],
                 [2, 3, 0, 0.5, 0, 0],
                 [2, 7, 0, 0.5, 0, 0],
                 [2, 10, 0, 0.5, 0, 0],
                 ]

    draw_nx_display_tree(actions_lst, dataset_number=0)
