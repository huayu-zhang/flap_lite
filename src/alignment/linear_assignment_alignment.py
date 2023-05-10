from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import numpy as np

from src.utils import tokens_to_span_classes, tokenize_default
from src.utils import Span


def string_to_2d_array(s):
    return np.array(list(s)).reshape(-1, 1)


def bin_dist(x, y):
    return int(x == y)


class LinearAssignmentAlignment:

    def __init__(self, seq1, seq2):
        self.seq1 = seq1
        self.seq2 = seq2

        self.cost_matrix = None
        self.assignment_mapping = None
        self.diag_paths = None

    def _prepare_cost_matrix(self, diag_bonus=True, token_bonus=True):
        a1 = string_to_2d_array(self.seq1)
        a2 = string_to_2d_array(self.seq2)

        self.cost_matrix = cdist(a1, a2, metric=bin_dist)

        if diag_bonus:
            self._apply_diag_bonus()

        if token_bonus:
            self._apply_token_bonus()

    def _find_diag_paths(self):
        x_map = self.cost_matrix.copy()

        paths = []

        n_row = x_map.shape[0]
        n_col = x_map.shape[1]

        for i in range(n_row):
            for j in range(n_col):

                if x_map[i, j]:

                    start = (i, j)
                    pi, pj = start

                    try:
                        while x_map[pi, pj]:
                            x_map[pi, pj] = 0
                            pi += 1
                            pj += 1
                    except IndexError:
                        pi -= 1
                        pj -= 1
                        end = (pi, pj)
                        x_map[pi, pj] = 0
                    else:
                        pi -= 1
                        pj -= 1
                        end = (pi, pj)
                        x_map[pi, pj] = 0

                    paths.append([start, end])

        self.diag_paths = paths

    def _apply_diag_bonus(self):

        if self.diag_paths is None:
            self._find_diag_paths()

        for path in self.diag_paths:

            (i, j), (i_end, j_end) = path
            len_path_bonus = i_end - i

            while (i <= i_end) and (j <= j_end):
                self.cost_matrix[i, j] += len_path_bonus

                i += 1
                j += 1

    def _apply_token_bonus(self, tokenizer=tokenize_default):

        if self.diag_paths is None:
            self._find_diag_paths()

        i_spans = []
        j_spans = []

        for path in self.diag_paths:
            (i0, j0), (i1, j1) = path
            i_spans.append(Span(i0, i1))
            j_spans.append(Span(j0, j1))

        t1 = tokenizer(self.seq1)
        t2 = tokenizer(self.seq2)

        sp1 = tokens_to_span_classes(t1)
        sp2 = tokens_to_span_classes(t2)

        for span in sp1:
            if any([span in i_span for i_span in i_spans]):
                self.cost_matrix[span.start:span.end + 1, :] *= 2

        for span in sp2:
            if any([span in j_span for j_span in j_spans]):
                self.cost_matrix[:, span.start:span.end + 1] *= 2

    def diag_paths_to_spans(self):
        i_spans = []
        j_spans = []

        for path in self.diag_paths:
            (i0, j0), (i1, j1) = path
            i_spans.append(Span(i0, i1))
            j_spans.append(Span(j0, j1))

        return i_spans, j_spans

    def _align(self):
        if self.cost_matrix is None:
            self._prepare_cost_matrix()

        self.assignment_mapping = [(i, j) for i, j in
                                   np.array(linear_sum_assignment(self.cost_matrix, maximize=True)).transpose()
                                   if self.seq1[i] == self.seq2[j]]

    def get_result(self):

        if self.assignment_mapping is None:
            self._align()

        return NonSequentialAlignmentResult(self.assignment_mapping, self.seq1, self.seq2)


class NonSequentialAlignmentResult:

    def __init__(self, mapping, seq1, seq2):
        self.mapping = mapping
        self.seq1 = seq1
        self.seq2 = seq2

        self.meta = dict(mapping=mapping,
                         n_aligned=len(mapping),
                         len_x=len(seq1),
                         len_y=len(seq2))

    def __str__(self):
        mapping = self.meta['mapping']

        mapping_dict_x_to_y = dict(mapping)
        mapping_dict_y_to_x = {v: k for k, v in dict(mapping).items()}

        seq1 = self.seq1
        seq2 = self.seq2

        lines = []

        rx = seq1

        ry = ''.join(
            [seq2[mapping_dict_x_to_y[i]] if i in mapping_dict_x_to_y.keys() else '_' for i in range(len(seq1))]
        )

        lines.extend(['On Seq 1:', rx, ry])

        rx = ''.join(
            [seq1[mapping_dict_y_to_x[i]] if i in mapping_dict_y_to_x.keys() else '_' for i in range(len(seq2))])
        ry = seq2

        lines.extend(['On Seq 2:', rx, ry])

        return '\n'.join(lines)

    def __repr__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__

        return '\n'.join([f'<{module}.{qualname} object at {hex(id(self))}>',
                          str(self)])

    def count_insertion(self):

        mapping = self.meta['mapping']

        mapping_ends = [[min(mapping, key=lambda x: x[0])[0], max(mapping, key=lambda x: x[0])[0]],
                        [min(mapping, key=lambda x: x[1])[1], max(mapping, key=lambda x: x[1])[1]]]

        if len(self.meta['mapping']):
            n_insertion = 0

            for i in range(2):
                n_insertion += mapping_ends[i][1] - mapping_ends[i][0] - len(mapping) + 1

        else:
            n_insertion = 0

        self.meta['n_insertion'] = n_insertion

    def calculate_score(self):

        if 'n_insertion' not in self.meta:
            self.count_insertion()

        n_aligned = self.meta['n_aligned']
        len_x = self.meta['len_x']
        len_y = self.meta['len_y']

        try:
            perc_x = n_aligned/len_x
            perc_y = n_aligned/len_y
            f1 = 2 * perc_x * perc_y / (perc_x + perc_y)

        except ZeroDivisionError:
            perc_x = 0
            perc_y = 0
            f1 = 0

        self.meta['perc_x'] = perc_x
        self.meta['perc_y'] = perc_y
        self.meta['f1'] = f1

        self.meta['score'] = self.meta['f1']
