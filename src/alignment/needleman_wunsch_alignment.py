"""
Created on Wed Jun 16 22:45:43 2021

@author: hzhang3410
"""

import numpy as np
from enum import IntFlag
import copy

import re

from src.utils import tokens_to_spans, tokenize_default, f1, repl_positions
from src.distance import letter_freq_cosine_sim


class SequentialAlignmentResult:

    def __init__(self, alignment_result, insertion_label='_', name=None):
        self.alignment_result = alignment_result

        self.meta = dict(name=name)

        self.insertion_label = insertion_label

    def __str__(self):
        return '\n'.join([str(x) for x in self.alignment_result])

    def __repr__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__

        return '\n'.join([f'<{module}.{qualname} object at {hex(id(self))}>',
                          str(self)])

    def copy(self):
        return copy.deepcopy(self)

    def _prepare_mapping(self):

        rx, ry = self.alignment_result

        ix = 0
        iy = 0

        n_space_match = 0

        mapping = []

        for i in range(len(rx)):
            if rx[i] == ry[i]:
                mapping.append([ix, iy])
                ix += 1
                iy += 1

                if rx[i] == ' ':
                    n_space_match += 1

            elif rx[i] == '_':
                iy += 1
            elif ry[i] == '_':
                ix += 1
            else:
                ix += 1
                iy += 1

        self.meta['mapping'] = mapping
        self.meta['n_aligned'] = len(mapping)
        self.meta['len_x'] = ix
        self.meta['len_y'] = iy
        self.meta['n_space_match'] = n_space_match

    def count_insertion(self):

        if 'mapping' not in self.meta:
            self._prepare_mapping()

        mapping = self.meta['mapping']

        if len(self.meta['mapping']):

            mapping_ends = [[min(mapping, key=lambda x: x[0])[0], max(mapping, key=lambda x: x[0])[0]],
                            [min(mapping, key=lambda x: x[1])[1], max(mapping, key=lambda x: x[1])[1]]]

            n_insertion = 0

            for i in range(2):
                n_insertion += mapping_ends[i][1] - mapping_ends[i][0] - len(mapping) + 1

        else:
            n_insertion = 0

        self.meta['n_insertion'] = n_insertion

    def calculate_letter_freq_sim(self):
        rx, ry = self.alignment_result

        self.meta['letter_freq_sim'] = letter_freq_cosine_sim(rx, ry)

    def calculate_score(self):

        if 'mapping' not in self.meta:
            self._prepare_mapping()

        if 'n_insertion' not in self.meta:
            self.count_insertion()

        if 'letter_freq_sim' not in self.meta:
            self.calculate_letter_freq_sim()

        n_aligned = self.meta['n_aligned']
        len_x = self.meta['len_x']
        len_y = self.meta['len_y']

        try:
            perc_x = n_aligned / len_x
            perc_y = n_aligned / len_y
            f1 = 2 * perc_x * perc_y / (perc_x + perc_y)

        except ZeroDivisionError:
            perc_x = 0
            perc_y = 0
            f1 = 0

        self.meta['perc_x'] = perc_x
        self.meta['perc_y'] = perc_y
        self.meta['f1'] = f1

        self.meta['score'] = self.meta['f1'] + 0.1 * self.meta['letter_freq_sim'] \
            - 0.01 * self.meta['n_insertion'] - 0.01 * self.meta['n_space_match']

        rx, ry = self.alignment_result

        self.meta['residual_x'] = ''.join([rx[i] for i in range(len(rx)) if ry[i] == '_'])
        self.meta['residual_y'] = ''.join([ry[i] for i in range(len(ry)) if rx[i] == '_'])

    def anneal_to_splitter(self):
        rx, ry = self.alignment_result

        sy = tokens_to_spans(tokenize_default(ry))

        rx = list(rx)
        ry = list(ry)

        p_inword_insertion = []

        for start, end in sy:

            n_match = sum([x == y for x, y in zip(rx[start:end], ry[start:end])])
            real_length = end - start - sum([x == '_' for x in ry[start:end]])

            if (n_match == 0) or (n_match == real_length):
                pass

            elif n_match / real_length < 0.80:
                rx[start:end] = ['_' if x == y else x for x, y in zip(rx[start:end], ry[start:end])]

            else:
                rx[start:end] = ry[start:end]
                p_inword_insertion.extend([i for i in range(start, end) if ry[i] == '_'])

        for i in sorted(p_inword_insertion, reverse=True):
            del rx[i]
            del ry[i]

        self.alignment_result = (''.join(rx), ''.join(ry))

    def get_summary(self):
        return {'alignment_result': self.alignment_result,
                'meta': self.meta}


class SequentialAlignmentResults:

    def __init__(self, list_of_alignment_results):
        self.alignment_results = list_of_alignment_results
        self.aggregated_alignment_score = None
        self.alignment_features = None

    def __str__(self):
        return "\n".join([str(ar) for ar in self.alignment_results])

    def __repr__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__

        return '\n'.join([f'<{module}.{qualname} object at {hex(id(self))}>',
                          str(self)])

    def copy(self):
        return copy.deepcopy(self)

    def append(self, ar):
        assert isinstance(ar, SequentialAlignmentResult), 'Can only append class SequentialAlignmenbtResult'

        self.alignment_results.append(ar)

    def min_insertion_alignment(self, selection='last'):

        if len(self.alignment_results) == 1:
            return self.alignment_results[0]
        else:
            for ar in self.alignment_results:
                ar.count_insertion()

        min_n_insertion = min([ar.meta['n_insertion'] for ar in self.alignment_results])

        selected_alignment = [ar for ar in self.alignment_results if ar.meta['n_insertion'] == min_n_insertion]

        if selection == 'last':
            return selected_alignment[-1]
        elif selection == 'first':
            return selected_alignment[0]

    def max_score_alignment(self, selection='last'):

        if len(self.alignment_results) == 1:
            return self.alignment_results[0]
        else:
            for ar in self.alignment_results:
                ar.calculate_score()

        max_score = max([ar.meta['score'] for ar in self.alignment_results])

        selected_alignment = [ar for ar in self.alignment_results if ar.meta['score'] == max_score]

        if selection == 'last':
            return selected_alignment[-1]
        elif selection == 'first':
            return selected_alignment[0]

    def score_alignments_by_rule(self):

        selected_ars = [ar for ar in self.alignment_results if ar.meta['score']]
        all_ars = [ar for ar in self.alignment_results]

        if sum([ar.meta['score'] for ar in selected_ars]) > 0:

            alignment_score = {
                'mean_letter_freq_sim': np.mean([ar.meta['letter_freq_sim']
                                                 for ar in selected_ars]),
                'total_perc_uprn': sum([np.square(ar.meta['perc_x']) * ar.meta['len_x']
                                        for ar in selected_ars]) / sum([ar.meta['len_x'] for ar in selected_ars]),
                'total_perc_address': sum([ar.meta['perc_y'] for ar in selected_ars]),
            }

            if sum([ar.meta['len_x'] for ar in selected_ars]) > 51:
                alignment_score['total_perc_uprn'] *= 1.1

            alignment_score['total_f1'] = \
                f1(alignment_score['total_perc_uprn'],
                   alignment_score['total_perc_address'])

            end_residual = repl_positions(selected_ars[-1].alignment_result[1].replace('_', ''),
                                          [py for _, py in selected_ars[-1].meta['mapping']])

            alignment_score['number_bonus'] = 0.3 if re.search(pattern=r'\d+',
                                                               string=end_residual[:-8]) is None else 0
            alignment_score['single_char_bonus'] = 0.1 if re.search(pattern=r'(\W|\d|^|_)[A-Z](\W|\d|$|_)',
                                                                    string=end_residual[:-8]) is None else 0

            x_residual = ' '.join([ar.meta['residual_x'] for ar in all_ars if ar.meta['name'] != 'POSTCODE'])

            alignment_score['uprn_number_penalty'] = -0.3 if re.search(pattern=r'\d+',
                                                                       string=x_residual) is not None else 0
            alignment_score['uprn_single_char_penalty'] = -0.1 if re.search(pattern=r'(\W|\d|^|_)[A-Z](\W|\d|$|_)',
                                                                            string=x_residual) is not None else 0

            alignment_score['total_insertion'] = sum([ar.meta['n_insertion'] for ar in selected_ars])

            alignment_score['total_score'] = \
                alignment_score['total_f1'] + \
                0.1 * alignment_score['mean_letter_freq_sim'] + \
                alignment_score['number_bonus'] + \
                alignment_score['single_char_bonus'] + \
                alignment_score['uprn_number_penalty'] + \
                alignment_score['uprn_single_char_penalty'] - \
                0.01 * alignment_score['total_insertion']

        else:
            alignment_score = {
                'total_score': 0
            }

        self.aggregated_alignment_score = alignment_score

    def score_alignments_by_model(self, model):

        feature_dict = self.get_alignment_features()

        x = np.fromiter(feature_dict.values(), dtype=float).reshape(1, -1)

        self.aggregated_alignment_score['clf_score'] = model.predict_proba(x)[0, 1]

    def get_alignment_features(self, header):

        summary = dict()

        name_index = 0

        for ar in self.alignment_results:
            for metric in ['letter_freq_sim', 'n_insertion', 'perc_x', 'perc_y']:
                if ar.meta['name'] is None:
                    ar.meta['name'] = name_index
                    name_index += 1

                summary['%s_%s' % (ar.meta['name'], metric)] = ar.meta[metric]

        if self.aggregated_alignment_score is None:
            self.score_alignments_by_rule()

        summary.update(self.aggregated_alignment_score)

        if 'clf_score' in summary:
            del summary['clf_score']

        normalized_features = {}

        for k in header:
            if k in summary:
                normalized_features[k] = summary[k]
            else:
                normalized_features[k] = 0

        self.alignment_features = normalized_features

        return self.alignment_features


class Direction(IntFlag):
    """
    Enum representing directions
    """

    UP = 1
    LEFT = 2
    DIAG = 4


class Cell:
    """
    Class representing cell of the matrix with value and directions
    """

    def __init__(self):
        self.value = None
        self.directions = None

    def __str__(self):
        value = "None" if self.value is None else str(self.value)
        direction = "None" if self.directions is None else str(self.directions)
        return f"Value: {value}, Direction: {direction}"

    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.value == other.value and self.directions == other.directions
        return False


class NeedlemanWunschAlignment:
    """
    Class representing Needleman-Wunsch algorithm
    """

    def __init__(self, seq1, seq2,
                 match=1, mismatch=-1, gap=0,
                 insertion_label='_'):
        self.seq1 = "-" + seq1
        self.seq2 = "-" + seq2
        self.len1 = len(seq1)
        self.len2 = len(seq2)
        self.config = {
            'SAME': match,
            'DIFF': mismatch,
            'GAP_PENALTY': gap,
            'MAX_NUMBER_PATHS': 20,
            'MAX_SEQ_LENGTH': 2000
        }
        self.insertion_label = insertion_label
        self._scoring_matrix = None

    def _prepare_scoring_matrix(self):
        # Scoring matrix dimensions
        n_row = self.len2 + 1
        n_col = self.len1 + 1

        scoring_matrix = np.empty((n_row, n_col), dtype=Cell)

        scoring_matrix[0, 0] = Cell()
        scoring_matrix[0, 0].value = 0
        scoring_matrix[0, 0].directions = None

        for j in range(1, n_col):
            scoring_matrix[0, j] = Cell()
            scoring_matrix[0, j].value = j * self.config['GAP_PENALTY']
            scoring_matrix[0, j].directions = Direction.LEFT

        for i in range(1, n_row):
            scoring_matrix[i, 0] = Cell()
            scoring_matrix[i, 0].value = i * self.config['GAP_PENALTY']
            scoring_matrix[i, 0].directions = Direction.UP

            for j in range(1, n_col):
                val_up = scoring_matrix[i - 1, j].value + self.config['GAP_PENALTY']
                val_left = scoring_matrix[i, j - 1].value + self.config['GAP_PENALTY']
                val_diag = scoring_matrix[i - 1, j - 1].value + (
                    self.config['SAME'] if self.seq1[j] == self.seq2[i] else self.config['DIFF'])
                values = {Direction.UP: val_up, Direction.LEFT: val_left, Direction.DIAG: val_diag}

                scoring_matrix[i, j] = self._get_cell(values)

        return scoring_matrix

    def _get_cell(self, values):
        max_value = max(values.values())

        directions_list = [idx for idx, val in values.items() if val == max_value]

        direction = 0
        for d in directions_list:
            direction = direction | d

        cell = Cell()
        cell.value = max_value
        cell.directions = direction

        return cell

    def _next_step(self, char, main_direction, directions):
        value = directions.value
        if value & Direction.UP.value:
            yield char if main_direction == Direction.UP else self.insertion_label, (0, -1)
        if value & Direction.LEFT.value:
            yield char if main_direction == Direction.LEFT else self.insertion_label, (-1, 0)
        if value & Direction.DIAG.value:
            yield char, (-1, -1)

    def _get_result(self, x, y, R, seq, direction):
        if x == 0 and y == 0:
            yield R
            return

        cell = self.get_scoring_matrix()[x, y]
        char = seq[y if direction == Direction.LEFT else x]

        for letter, coord in self._next_step(char, direction, cell.directions):
            new_R = letter + R
            new_x = x + coord[1]
            new_y = y + coord[0]
            for result in self._get_result(new_x, new_y, new_R, seq, direction):
                yield result

    def get_result_raw(self):
        gen1 = self._get_result(self.len2, self.len1, "", self.seq1, Direction.LEFT)
        gen2 = self._get_result(self.len2, self.len1, "", self.seq2, Direction.UP)
        try:
            for i in range(0, self.config['MAX_NUMBER_PATHS']):
                yield (next(gen1), next(gen2))
        except StopIteration:
            return

    def get_result(self):
        result_list = [SequentialAlignmentResult(a, self.insertion_label) for a in list(self.get_result_raw())]
        return SequentialAlignmentResults(result_list)

    def get_scoring_matrix(self):
        if self._scoring_matrix is None:
            self._scoring_matrix = self._prepare_scoring_matrix()
        return self._scoring_matrix

    def get_score(self):
        return self.get_scoring_matrix()[self.len2, self.len1].value

    def print_results(self, output):
        with open(output, "w") as out:
            out.write(f"SCORE={self.get_score()} \n")
            result = self.get_result_raw()
            result_list = list(result)
            for r in result_list:
                out.write("\n")
                out.write(f"{r[0]}\n")
                out.write(f"{r[1]}\n")


class NeedlemanWunschAlignmentToken:
    """
    Class representing Needleman-Wunsch algorithm for tokens
    """

    @staticmethod
    def _token_align_to_string_align(alignment_result):
        rx_token, ry_token = alignment_result.alignment_result
        insertion_label = alignment_result.insertion_label

        rx = ''
        ry = ''

        for i in range(len(rx_token)):
            if rx_token[i] == ry_token[i]:
                rx += rx_token[i]
                ry += ry_token[i]
            elif rx_token[i] == insertion_label:
                rx += insertion_label * len(ry_token[i])
                ry += ry_token[i]
            elif ry_token[i] == insertion_label:
                rx += rx_token[i]
                ry += insertion_label * len(rx_token[i])
            else:
                if len(rx_token[i]) == len(ry_token[i]):
                    rx += rx_token[i]
                    ry += ry_token[i]
                else:
                    nw_temp_result = list(NeedlemanWunschAlignment(rx_token[i], ry_token[i]).get_result_raw())[0]
                    rx_temp, ry_temp = nw_temp_result
                    rx += rx_temp
                    ry += ry_temp

        return SequentialAlignmentResult((rx, ry), insertion_label)

    def __init__(self, seq1, seq2,
                 match=1, mismatch=-1, gap=0,
                 allow_fuzzy=False, dist_funct=None, dist_tol=None,
                 insertion_label='_'):
        self.seq1 = ["-"] + seq1
        self.seq2 = ["-"] + seq2
        self.len1 = len(seq1)
        self.len2 = len(seq2)
        self.config = {
            'SAME': match,
            'DIFF': mismatch,
            'GAP_PENALTY': gap,
            'MAX_NUMBER_PATHS': 20,
            'MAX_SEQ_LENGTH': 2000
        }
        self.insertion_label = insertion_label
        self.allow_fuzzy = allow_fuzzy
        self.dist_funct = dist_funct
        self.dist_tol = dist_tol
        self._scoring_matrix = None

    def _prepare_scoring_matrix(self):
        # Scoring matrix dimensions
        n_row = self.len2 + 1
        n_col = self.len1 + 1

        scoring_matrix = np.empty((n_row, n_col), dtype=Cell)

        scoring_matrix[0, 0] = Cell()
        scoring_matrix[0, 0].value = 0
        scoring_matrix[0, 0].directions = None

        for j in range(1, n_col):
            scoring_matrix[0, j] = Cell()
            scoring_matrix[0, j].value = j * self.config['GAP_PENALTY']
            scoring_matrix[0, j].directions = Direction.LEFT

        for i in range(1, n_row):
            scoring_matrix[i, 0] = Cell()
            scoring_matrix[i, 0].value = i * self.config['GAP_PENALTY']
            scoring_matrix[i, 0].directions = Direction.UP

            for j in range(1, n_col):
                val_up = scoring_matrix[i - 1, j].value + self.config['GAP_PENALTY']
                val_left = scoring_matrix[i, j - 1].value + self.config['GAP_PENALTY']
                val_diag = scoring_matrix[i - 1, j - 1].value + (
                    self.config['SAME'] if self.seq1[j] == self.seq2[i] else self.config['DIFF'])
                values = {Direction.UP: val_up, Direction.LEFT: val_left, Direction.DIAG: val_diag}

                scoring_matrix[i, j] = self._get_cell(values)

        return scoring_matrix

    def _prepare_scoring_matrix_fuzzy(self):
        # Scoring matrix dimensions
        n_row = self.len2 + 1
        n_col = self.len1 + 1

        scoring_matrix = np.empty((n_row, n_col), dtype=Cell)

        scoring_matrix[0, 0] = Cell()
        scoring_matrix[0, 0].value = 0
        scoring_matrix[0, 0].directions = None

        for j in range(1, n_col):
            scoring_matrix[0, j] = Cell()
            scoring_matrix[0, j].value = j * self.config['GAP_PENALTY']
            scoring_matrix[0, j].directions = Direction.LEFT

        for i in range(1, n_row):
            scoring_matrix[i, 0] = Cell()
            scoring_matrix[i, 0].value = i * self.config['GAP_PENALTY']
            scoring_matrix[i, 0].directions = Direction.UP

            for j in range(1, n_col):
                val_up = scoring_matrix[i - 1, j].value + self.config['GAP_PENALTY']
                val_left = scoring_matrix[i, j - 1].value + self.config['GAP_PENALTY']
                val_diag = scoring_matrix[i - 1, j - 1].value + (
                    self.config['SAME'] if self.dist_funct(self.seq1[j], self.seq2[i]) < self.dist_tol else self.config[
                        'DIFF'])
                values = {Direction.UP: val_up, Direction.LEFT: val_left, Direction.DIAG: val_diag}

                scoring_matrix[i, j] = self._get_cell(values)

        return scoring_matrix

    def _get_cell(self, values):
        max_value = max(values.values())

        directions_list = [idx for idx, val in values.items() if val == max_value]

        direction = 0
        for d in directions_list:
            direction = direction | d

        cell = Cell()
        cell.value = max_value
        cell.directions = direction

        return cell

    def _next_step(self, char, main_direction, directions):
        value = directions.value
        if value & Direction.UP.value:
            yield char if main_direction == Direction.UP else self.insertion_label, (0, -1)
        if value & Direction.LEFT.value:
            yield char if main_direction == Direction.LEFT else self.insertion_label, (-1, 0)
        if value & Direction.DIAG.value:
            yield char, (-1, -1)

    def _get_result(self, x, y, R, seq, direction):
        if x == 0 and y == 0:
            yield R
            return

        cell = self.get_scoring_matrix()[x, y]
        char = seq[y if direction == Direction.LEFT else x]

        for letter, coord in self._next_step(char, direction, cell.directions):
            new_R = [letter] + R
            new_x = x + coord[1]
            new_y = y + coord[0]
            for result in self._get_result(new_x, new_y, new_R, seq, direction):
                yield result

    def get_result_raw(self):
        gen1 = self._get_result(self.len2, self.len1, [], self.seq1, Direction.LEFT)
        gen2 = self._get_result(self.len2, self.len1, [], self.seq2, Direction.UP)
        try:
            for i in range(0, self.config['MAX_NUMBER_PATHS']):
                yield next(gen1), next(gen2)
        except StopIteration:
            return

    def get_result(self):
        result_list = [self._token_align_to_string_align(SequentialAlignmentResult(a, self.insertion_label))
                       for a in list(self.get_result_raw())]
        return SequentialAlignmentResults(result_list)

    def get_scoring_matrix(self):
        if self._scoring_matrix is None:
            if self.allow_fuzzy:
                self._scoring_matrix = self._prepare_scoring_matrix_fuzzy()
            else:
                self._scoring_matrix = self._prepare_scoring_matrix()
        return self._scoring_matrix

    def get_score(self):
        return self.get_scoring_matrix()[self.len2, self.len1].value

    def print_results(self, output):
        with open(output, "w") as out:
            out.write(f"SCORE={self.get_score()} \n")
            result = self.get_result_raw()
            result_list = list(result)
            for r in result_list:
                out.write("\n")
                out.write(f"{r[0]}\n")
                out.write(f"{r[1]}\n")
