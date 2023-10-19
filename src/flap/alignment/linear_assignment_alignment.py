from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import numpy as np

from flap.utils import tokens_to_span_classes, tokenize_default
from flap.utils import Span


def string_to_2d_array(s):
    return np.array(list(s)).reshape(-1, 1)


def bin_dist(x, y):
    return int(x == y)


class LinearAssignmentAlignment:

    def __init__(self, seq1, seq2):
        self.seq1 = seq1

        self.seq2_index = {}

        if isinstance(seq2, str):
            self.seq2 = seq2
        elif isinstance(seq2, dict):
            tmp_start = 0
            self.seq2 = ' '.join([v for v in seq2.values() if len(v)])

            for k, v in seq2.items():

                if len(v):
                    self.seq2_index[k] = Span(tmp_start, tmp_start + len(v))
                    tmp_start += len(v) + 1
                else:
                    self.seq2_index[k] = []

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

        return NonSequentialAlignmentResult(self.assignment_mapping, self.seq1, self.seq2, self.seq2_index)


class NonSequentialAlignmentResult:

    def __init__(self, mapping, seq1, seq2, seq2_index=None):
        self.mapping = mapping
        self.seq1 = seq1
        self.seq2 = seq2
        self.seq2_index = seq2_index

        self.meta = dict(mapping=mapping,
                         n_aligned=len(mapping),
                         ft_len_x=len(seq1),
                         ft_len_y=len(seq2))

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

        try:
            mapping_ends = [[min(mapping, key=lambda x: x[0])[0], max(mapping, key=lambda x: x[0])[0]],
                            [min(mapping, key=lambda x: x[1])[1], max(mapping, key=lambda x: x[1])[1]]]
        except ValueError:
            self.meta['ft_n_insertion'] = 0
            return

        if len(self.meta['mapping']):
            n_insertion = 0

            for i in range(2):
                n_insertion += mapping_ends[i][1] - mapping_ends[i][0] - len(mapping) + 1

        else:
            n_insertion = 0

        self.meta['ft_n_insertion'] = n_insertion

    def calculate_subscore(self):

        seq2_index = self.seq2_index
        mapping = self.mapping

        seq2_backtrace = {k: [] for k in seq2_index}
        seq2_perc = {}

        for i, j in mapping:

            for k, span in seq2_index.items():

                if j in span:
                    seq2_backtrace[k].append(i)
                    break

        for k in seq2_backtrace:
            if len(seq2_backtrace[k]):
                seq2_perc['ft_perc_%s' % k] = len(seq2_backtrace[k]) / len(seq2_index[k])
            else:
                seq2_perc['ft_perc_%s' % k] = 0

        self.meta.update(seq2_perc)

    def calculate_score(self):

        if 'ft_n_insertion' not in self.meta:
            self.count_insertion()

        n_aligned = self.meta['n_aligned']
        len_x = self.meta['ft_len_x']
        len_y = self.meta['ft_len_y']

        try:
            perc_x = n_aligned/len_x
            perc_y = n_aligned/len_y
            f1 = 2 * perc_x * perc_y / (perc_x + perc_y)

        except ZeroDivisionError:
            perc_x = 0
            perc_y = 0
            f1 = 0

        self.meta['ft_perc_x'] = perc_x
        self.meta['ft_perc_y'] = perc_y
        self.meta['ft_f1'] = f1

        if self.seq2_index is not None:
            self.calculate_subscore()

        self.meta['score'] = self.meta['ft_f1']

    def get_score(self):

        self.calculate_score()
        score = {k: self.meta[k] for k in self.meta if 'ft_' in k}

        return score


#
# uprn = {'CHANGE_TYPE': 'I', 'UPRN': '10091986837', 'UDPRN': '8253456',
#         'ORGANISATION_NAME': 'MANSFIELD CARE ADMINISTRATION', 'DEPARTMENT_NAME': '', 'SUB_BUILDING_NAME': '',
#         'BUILDING_NAME': '', 'BUILDING_NUMBER': '99', 'DEPENDENT_THOROUGHFARE': '', 'THOROUGHFARE': 'CRAIGHALL ROAD',
#         'DOUBLE_DEPENDENT_LOCALITY': '', 'DEPENDENT_LOCALITY': '', 'POST_TOWN': 'EDINBURGH', 'POSTCODE': 'EH6 4RD',
#         'POSTCODE_TYPE': 'S', 'DELIVERY_POINT_SUFFIX': '1T', 'WELSH_DEPENDENT_THOROUGHFARE': '',
#         'WELSH_THOROUGHFARE': '', 'WELSH_DOUBLE_DEPENDENT_LOCALITY': '', 'WELSH_DEPENDENT_LOCALITY': '',
#         'WELSH_POST_TOWN': '', 'PO_BOX_NUMBER': '', 'PROCESS_DATE': '2018-11-06',
#         'START_DATE': '2018-12-06', 'END_DATE': '', 'LAST_UPDATE_DATE': '2018-12-06',
#         'ENTRY_DATE': '2012-03-19', 'GEOMETRY': None, 'n_tenement': '1', 'type_of_micro': '10001',
#         'type_of_macro': '0100', 'number_like_0': '99', 'number_like_1': '', 'number_like_2': '', 'number_like_3': '',
#         'number_like_4': '', 'pc0': 'EH6', 'pc1': '4RD', 'pc_area': 'EH', 'pc_district': '6', 'pc_sector': '4',
#         'pc_unit_0': 'R', 'pc_unit_1': 'D'}

#
# from src.database.sql import SqlDB
#
#
# sql_db = SqlDB('/home/huayu_ssh/PycharmProjects/dres_r/db/scotland_curl')
#
# columns = ['ORGANISATION_NAME', 'DEPARTMENT_NAME', 'SUB_BUILDING_NAME', 'BUILDING_NAME', 'BUILDING_NUMBER',
# 'DEPENDENT_THOROUGHFARE', 'THOROUGHFARE', 'DOUBLE_DEPENDENT_LOCALITY', 'DEPENDENT_LOCALITY', 'POST_TOWN', 'POSTCODE']
#
#
# res = sql_db.sql_query('select * from indexed limit 1')[0]
#
# d = {k: uprn[k] for k in columns}
#
# laa = LinearAssignmentAlignment(seq1='MANSFIELD CARE ADMINISTRATION 99 CRAIGHALL ROAD EDINBURGH EH6 4RD',
#                                 seq2=d)
#
# ar = laa.get_result()


#
# laa.seq2
# laa.seq2_index
#
#
# ar = laa.get_result()
#
#
# mapping = ar.mapping
#
# seq2 = ar.seq2
# seq2_index = ar.seq2_index
#
# seq2_backtrace = {k: [] for k in seq2_index}
# seq2_perc = {}
#
#
# for i, j in mapping:
#
#     for k, span in seq2_index.items():
#
#         if j in span:
#
#             seq2_backtrace[k].append(i)
#             break
#
# for k in seq2_backtrace:
#     if len(seq2_backtrace[k]):
#         seq2_perc[k] = len(seq2_backtrace[k])/len(seq2_index[k])
#     else:
#         seq2_perc[k] = 0
#
