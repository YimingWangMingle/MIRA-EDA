from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    op: str
    feature: int = -1
    value: int = -1
    delta: int = 0

    @property
    def family(self):
        if self.op.startswith("by_facet"):
            return "by_facet"
        if self.op.startswith("by_superset"):
            return "by_superset"
        if self.op.startswith("by_neighbors"):
            return "by_neighbors"
        if self.op == "by_distribution":
            return "by_distribution"
        return self.op

    @property
    def label(self):
        if self.op in {"by_facet_cont", "by_superset_cont", "by_neighbors_cont"}:
            return f"{self.op}:{self.feature}:{self.value}:{self.delta}"
        if self.op in {"by_facet_cover", "by_facet_wilderness", "by_facet_soil"}:
            return f"{self.op}:{self.value}"
        if self.op == "by_distribution":
            return f"{self.op}:{self.delta}"
        return self.op


def build_action_space(n_continuous=10, n_bins=10):
    actions = []

    for feature in range(n_continuous):
        for value in range(n_bins):
            actions.append(Action("by_facet_cont", feature=feature, value=value))
    for feature in range(n_continuous):
        actions.append(Action("by_superset_cont", feature=feature))
    for feature in range(n_continuous):
        actions.append(Action("by_neighbors_cont", feature=feature, delta=-1))
        actions.append(Action("by_neighbors_cont", feature=feature, delta=1))

    for value in range(1, 8):
        actions.append(Action("by_facet_cover", value=value))
    actions.append(Action("by_superset_cover"))

    for value in range(1, 5):
        actions.append(Action("by_facet_wilderness", value=value))
    actions.append(Action("by_superset_wilderness"))

    for value in range(1, 41):
        actions.append(Action("by_facet_soil", value=value))
    actions.append(Action("by_superset_soil"))


    max_distribution_step = max(1, int(n_bins) - 1)
    for delta in range(-max_distribution_step, max_distribution_step + 1):
        if delta != 0:
            actions.append(Action("by_distribution", delta=delta))
    return actions
