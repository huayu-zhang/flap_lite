"""
Consider split to different files later

"""

import pandas as pd
import numpy as np
from .cpu_count import available_cpu_count

# Span and token operations

class Span:

    def __init__(self, start, end):
        self.start, self.end = start, end
        assert self.start <= self.end, 'end < start is not allowed'

    def __contains__(self, item):

        if isinstance(item, type(self)):
            return (self.start <= item.start) and (self.end > item.end)
        elif isinstance(item, int) or isinstance(item, float) or isinstance(item, np.number):
            return (self.start <= item) and (self.end > item)
        else:
            raise TypeError('Can only compare a number or a Span')

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return 'Span: [%s, %s)' % (self.start, self.end)

    def __repr__(self):
        return 'Span(%s, %s)' % (self.start, self.end)


def tokens_to_span_classes(tokens):
    l_spans = []

    p = 0

    for token in tokens:
        l_spans.append(Span(p, p + len(token) - 1))

        p += len(token)

    return l_spans


def tokens_to_spans(tokens):
    l_spans = []

    p = 0

    for token in tokens:
        l_spans.append((p, p + len(token)))

        p += len(token)

    return l_spans


# Tokeniser

def tokenize_by_re(pattern, s):
    import re

    if isinstance(s, str):
        return [token for token in re.split(pattern, s) if len(token)]
    else:
        return ['non_str']


def tokenize_default(s):
    pattern = r'([^\w_]|\d+)'
    return tokenize_by_re(pattern, s)


# Math

def f1(prc, rec):
    return 2 * prc * rec / (prc + rec)


def f_beta(prc, rec, beta=2):
    return (1 + beta * beta) * prc * rec / (beta * beta * prc + rec)


# Utils

def parse_nodename(name):
    names = name.split(':')
    return names[0], names[1]


def span_to_lettermap(d_span, prior_l):
    for key in d_span:
        for span in d_span[key]:
            for i in range(*span):
                prior_l[i] = key

    return prior_l


def repl_positions(s, positions, repl='*'):
    sl = list(s)

    for p in positions:
        sl[p] = repl
    return ''.join(sl)


def repl_char_from_string(s, span, repl='*'):
    start, end = span

    return s[:start] + repl * (end - start) + s[end:]


def span_end(span):
    _, end = span
    return end


def dict_update_existingkeys(dict_to_update, new_dict):
    dict_to_update.update((k, new_dict[k]) for k in dict_to_update.keys())
    return dict_to_update


def span_to_string(span, s):
    start, end = span
    return s[start:end]


# UPRN formatting

def join_uprn_fields(d_uprn):
    # for key in d_uprn:
    #     if pd.isna(d_uprn[key]):
    #         d_uprn[key] = ''
    #     else:
    #         d_uprn[key] = str(d_uprn[key])

    return '\n'.join([
        ' | '.join([d_uprn['ORGANISATION_NAME'], d_uprn['DEPARTMENT_NAME'],
                    d_uprn['SUB_BUILDING_NAME'], d_uprn['BUILDING_NAME']]),
        ' | '.join([d_uprn['BUILDING_NUMBER'], d_uprn['DEPENDENT_THOROUGHFARE'], d_uprn['THOROUGHFARE']]),
        ' | '.join([d_uprn['DOUBLE_DEPENDENT_LOCALITY'], d_uprn['DEPENDENT_LOCALITY'],
                    d_uprn['POST_TOWN']]),
        ' | '.join([d_uprn['POSTCODE']])
    ])


def flatten(l):
    return [item for sublist in l for item in sublist]


def parse_uprn_fields(joined_fields):
    parsed = flatten(
              [ss.split(' | ') for ss in
               [s for s in joined_fields.split('\n')]]
                     )

    keys = [
        "ORGANISATION_NAME",
        "DEPARTMENT_NAME",
        "SUB_BUILDING_NAME",
        "BUILDING_NAME",
        "BUILDING_NUMBER",
        "DEPENDENT_THOROUGHFARE",
        "THOROUGHFARE",
        "DOUBLE_DEPENDENT_LOCALITY",
        "DEPENDENT_LOCALITY",
        "POST_TOWN",
        "POSTCODE"]

    return dict(zip(keys, parsed))


# File name


def timestamp_filename(filename):
    from pathlib import Path
    from time import ctime

    p = Path(filename)
    return "{0}-{2}{1}".format(Path.joinpath(p.parent, p.stem), p.suffix, ctime().replace(' ', '__').replace(':', '_'))
