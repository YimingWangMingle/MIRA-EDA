import re
import math
from typing import List

import pandas as pd

from atena.simulation.actions import Column

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


def add_tokens_to_set(tokens, tokens_set):
    for token in tokens:

        if not isinstance(token, str) and math.isnan(token):
            token = nan
        tokens_set.add(token)


def tokenize_column(df: pd.DataFrame, column: Column) -> List[str]:

    tokens_set = set()
    len_df = len(df)

    if len_df == 0:
        return []

    for idx, line in df[column].iteritems():
        line_tokens = tokenize_line_of_column(line, column)
        add_tokens_to_set(line_tokens, tokens_set)

    return list(tokens_set)
