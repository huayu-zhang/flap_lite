"""

Sql based match using combined queries (Need an evaluation for query efficiency)

Accelerate with indexing sql db

Accelerate with in-memory db

Linear alignment for each results

Avoid duplicated alignment using a memory dict

Use a classifier for early stop


Logic flow of matcher

require (parser object, db object, address, scorer, query priority, threshold score, max beam_width)

input (parsed address, db)

parse address ->

while < threshold and < max beam width

    query next rules

    update current beam width

    for row in query results

        match and score

get match of max score


To do:

Add functionality for multiplier addresses
    - [x] Index data
    - [ ] Index database
    - [x] Creating master file for scotland


Adding more rules and studying effectiveness of rules

How multipliers should be applied

If multiplier regex found in address

    If multiplier in UPRN
        Just match

    If multiplier not in UPRN
        guess the multiplier value
        convert the multiplier value to normal value

        get best match based on the model

Guess multiplier with UPRN information


"""


import os

import pandas as pd
import re
import numpy as np
import traceback
import pickle

from scipy.spatial.distance import pdist

import flap

from flap.database.sql import SqlDB
from flap.parser.rule_parser_fast import RuleParserFast
from flap.preprocessing import postcode_preproc, address_line_preproc

from flap.alignment.linear_assignment_alignment import LinearAssignmentAlignment
from flap.utils.progress_map import progress_map_tqdm_concurrent

# import time  # time

MODULE_PATH = os.path.dirname(flap.__file__)

try:
    DEFAULT_MODEL_PATH = [os.path.join(MODULE_PATH, 'model', path)
                          for path in os.listdir(os.path.join(MODULE_PATH, 'model')) if 'clf' in path][0]
except FileNotFoundError:
    DEFAULT_MODEL_PATH = None

DEFAULT_MULTIPLIER_INDEX = os.path.join(MODULE_PATH, 'parser', 'multiplier_index.csv')


class SqlMatcher:

    queries = [
        ['POSTCODE'],
        ['pc_area', 'pc_district', 'number_like_0'],
        ['POST_TOWN', 'THOROUGHFARE'],
        ['POST_TOWN', 'number_like_0']
    ]

    def __init__(self, sql_db, parser=None, scorer_path=None, multiplier_indices=None):
        self.sql_db = sql_db
        self.parser = parser
        self.multiplier_indices = multiplier_indices
        self.scorer = None

        if self.parser is None:
            self.parser = RuleParserFast(sql_db)

        if scorer_path is None:
            if DEFAULT_MODEL_PATH is not None:
                try:
                    self.scorer = ClassifierScorer(DEFAULT_MODEL_PATH)
                except FileNotFoundError:
                    pass
        else:
            self.scorer = ClassifierScorer(scorer_path)

        if multiplier_indices is None:
            try:
                self.multiplier_indices = pd.read_csv(DEFAULT_MULTIPLIER_INDEX, index_col=0)
            except FileNotFoundError:
                pass

        self.match_config = {
            'max_beam_width': 200,
            'score_threshold': 0.3,
            'troubleshoot': False
        }

        for query in self.queries:
            self.sql_db.create_index(table_name='indexed', columns=query)
            self.sql_db.create_index(table_name='expanded', columns=query)

        self.sql_db.create_index(table_name='indexed', columns=['UPRN'])
        self.sql_db.create_index(table_name='expanded', columns=['UPRN'])

#         self.t_temp = time.time()  # time
#         self.t_log = {}  # time

#     def _record_time(self, checkpoint_name):  # time
#         # time
#         self.t_log[checkpoint_name] = time.time() - self.t_temp  # time
#         self.t_temp = time.time()  # time

    def set_match_config(self, **config):
        for k, v in config.items():
            self.match_config[k] = v

    def _query(self, query, table_name, parsed):

        query_task = {k: parsed['FOR_QUERY'][k].replace("'", "''") for k in query}

        query_string = query_task_to_sql(query_task, table_name)

        res = pd.DataFrame(self.sql_db.sql_query(query_string), columns=self.sql_db.get_columns_of_table(table_name))

        return res

    @staticmethod
    def _match_one_record(parsed, uprn_row):

        uprn_prepared = prepare_uprn(uprn_row)

        number_like_matching_matrix = get_number_like_matching_matrix(list(parsed['NUMBER_LIKE'].values()),
                                                                      uprn_prepared['NUMBER_LIKE'])

        ar = LinearAssignmentAlignment(parsed['TEXTUAL'], uprn_prepared['TEXTUAL']).get_result()

        text_alignment_score = ar.get_score()

        try:
            postcode_matching_vector = postcode_matching(parsed['POSTCODE_SPLIT'], uprn_prepared['POSTCODE_SPLIT'])
        except KeyError:
            postcode_matching_vector = [0] * 5

        x = summarize_features(number_like_matching_matrix, text_alignment_score, postcode_matching_vector)

        return {'uprn_row': uprn_row, 'features': x}

    def generate_feature(self, address, uprn_row):

        address, _ = address_line_preproc(address)

        parsed = self.parser.parse(address, method='fast')

        res = self._match_one_record(parsed, uprn_row)

        return res['features']

    def _generate_feature_for_concurrent(self, row):

        address, _ = address_line_preproc(row['input_address'])

        parsed = self.parser.parse(address, method='fast')

        res = self._match_one_record(parsed, row)

        multi_regex = re.compile(r'(\d)[A-Z](\d+)')

        multi_regex_matches = {k: re.match(multi_regex, v) for k, v in parsed['NUMBER_LIKE'].items()}

        if any([multi_regex_matches[k] is not None for k in multi_regex_matches]):

            alt_address = address

            alt_parsed = parsed.copy()

            number_like = alt_parsed['NUMBER_LIKE']

            level = 1
            key = 'number_like_0'
            flat = 1

            list_of_indices = []

            for k, m in multi_regex_matches.items():
                if m is None:
                    list_of_indices.append(number_like[k])
                else:
                    level = int(m.group(1))
                    flat = int(m.group(2))
                    key = k
                    break

            if level == 1:
                alt_parsed['NUMBER_LIKE'][key] = str(flat)
                alt_address = re.sub(pattern=parsed['FOR_QUERY'][key], repl=alt_parsed['NUMBER_LIKE'][key],
                                     string=alt_address)

                alt_parsed['TEXTUAL'] = strip_number_like(alt_address)

                res_alt = self._match_one_record(alt_parsed, row)

                score = self.scorer.score(res['features'])
                score_alt = self.scorer.score(res_alt['features'])

                if score >= score_alt:
                    return res['features']
                else:
                    return res_alt['features']

            if len(list_of_indices) == 0:
                return res['features']

            if self.multiplier_indices is not None:

                max_level = 0
                max_flat = 0

                try:
                    index = '--'.join(['-'.join(list_of_indices), parsed['FOR_QUERY']['POSTCODE']])

                    if index in self.multiplier_indices.index:
                        max_level = self.multiplier_indices.loc[index, 'max_level']
                        max_flat = self.multiplier_indices.loc[index, 'max_flat']

                except KeyError:

                    pass

                try:
                    n_tenement = int(row['n_tenement'])
                except ValueError:
                    n_tenement = 1

                multiplier, _ = guess_multiplier(max_level, max_flat, n_tenement)
                alt_number = (level - 1) * multiplier + flat
                alt_parsed['NUMBER_LIKE'][key] = str(alt_number)

                alt_address = re.sub(pattern=parsed['FOR_QUERY'][key], repl=alt_parsed['NUMBER_LIKE'][key],
                                     string=alt_address)
                alt_parsed['TEXTUAL'] = strip_number_like(alt_address)

                res_alt = self._match_one_record(alt_parsed, row)

                score = self.scorer.score(res['features'])
                score_alt = self.scorer.score(res_alt['features'])

                if score >= score_alt:
                    return res['features']
                else:
                    return res_alt['features']

        return res['features']

    def score_matching_of_batch(self, df_batch, input_address_col='input_address', uprn_col='uprn',
                                max_workers=None, chunksize=None):

        df_batch['flap_eval_score'] = np.NaN

        df_batch_with_uprn = df_batch[~df_batch[uprn_col].isna()]
        df_batch_without_uprn = df_batch[df_batch[uprn_col].isna()]

        uprn_list = [s for s in df_batch_with_uprn[uprn_col].to_list() if isinstance(s, str)]

        res = self.sql_db.sql_query_by_column_values(table_name='indexed', column='UPRN', value_list=uprn_list)
        columns = self.sql_db.get_columns_of_table(table_name='indexed')
        res_df = pd.DataFrame.from_records(res, columns=columns)

        merge_df = pd.merge(left=df_batch_with_uprn, right=res_df, how='left', left_on=uprn_col, right_on='UPRN')

        if sum(~merge_df.UPRN.isna()) == 0:

            return df_batch

        merge_df_with_uprn = merge_df[~merge_df.UPRN.isna()].copy()
        merge_df_without_uprn = merge_df[merge_df.UPRN.isna()].copy()

        if max_workers == 1:
            X = [self.generate_feature(row[input_address_col], row)
                 for _, row in merge_df_with_uprn.iterrows()]

        else:
            X = progress_map_tqdm_concurrent(self._generate_feature_for_concurrent,
                                             (row for _, row in merge_df_with_uprn.iterrows()),
                                             max_workers=max_workers, chunksize=chunksize)

        self.scorer.estimator.set_params(n_jobs=max_workers)

        scores = self.scorer.score_batch(X)

        # Re-attached the missing values

        merge_df_with_uprn.loc[:, 'flap_eval_score'] = scores

        return pd.concat([merge_df_with_uprn, merge_df_without_uprn, df_batch_without_uprn], join='inner')

    def match(self, address):

#         self.t_temp = time.time()  # time
#         self._record_time('start')  # time

        address, log = address_line_preproc(address)

        if not log['proceed']:
            return {'score': 0}

        max_beam_width = self.match_config['max_beam_width']
        score_threshold = self.match_config['score_threshold']
        troubleshoot = self.match_config['troubleshoot']

#         self._record_time('preproc')  # time

        parsed = self.parser.parse(address, method='fast')

#         self._record_time('fast_parse')  # time

        current_bw = 0

        stop_loop = False

        best_res = None

        results = []

        for table_name in ['indexed', 'expanded']:
            # Start of main loop on Tables

            row_matched = []

            for query in self.queries:

                # Start of Loop Over queries

                try:
                    res = self._query(query, table_name, parsed)

                except KeyError:
                    parsed = self.parser.parse(address, method='all')
                    try:
                        res = self._query(query, table_name, parsed)
                    except KeyError:
                        continue

                current_bw += len(res)

                # Matching loop

                for i, row in res.iterrows():
                    row_d = row.to_dict()

                    if row_d not in row_matched:
                        results.append(self._match_one_record(parsed, row))
                        row_matched.append(row_d)

                # Deal with multiplier

                multi_regex = re.compile(r'(\d)[A-Z](\d+)')

                multi_regex_matches = {k: re.match(multi_regex, v) for k, v in parsed['NUMBER_LIKE'].items()}

                if any([multi_regex_matches[k] is not None for k in multi_regex_matches]):

                    alt_address = address

                    alt_parsed = parsed.copy()

                    number_like = alt_parsed['NUMBER_LIKE']

                    level = 1
                    key = 'number_like_0'
                    flat = 1

                    list_of_indices = []

                    for k, m in multi_regex_matches.items():
                        if m is None:
                            list_of_indices.append(number_like[k])
                        else:
                            level = int(m.group(1))
                            flat = int(m.group(2))
                            key = k
                            break

                    if level == 1:
                        alt_parsed['NUMBER_LIKE'][key] = str(flat)
                        alt_address = re.sub(pattern=parsed['FOR_QUERY'][key], repl=alt_parsed['NUMBER_LIKE'][key],
                                             string=alt_address)

                        alt_parsed['TEXTUAL'] = strip_number_like(alt_address)

                        for i, row in res.iterrows():
                            results.append(self._match_one_record(alt_parsed, row))

                        continue

                    if len(list_of_indices) == 0:
                        continue

                    if self.multiplier_indices is not None:

                        max_level = 0
                        max_flat = 0

                        try:
                            index = '--'.join(['-'.join(list_of_indices), parsed['FOR_QUERY']['POSTCODE']])

                            if index in self.multiplier_indices.index:
                                max_level = self.multiplier_indices.loc[index, 'max_level']
                                max_flat = self.multiplier_indices.loc[index, 'max_flat']

                        except KeyError:

                            pass

                        for i, row in res.iterrows():

                            try:
                                n_tenement = int(row['n_tenement'])
                            except ValueError:
                                n_tenement = 1

                            multiplier, _ = guess_multiplier(max_level, max_flat, n_tenement)
                            alt_number = (level - 1) * multiplier + flat
                            alt_parsed['NUMBER_LIKE'][key] = str(alt_number)

                            alt_address = re.sub(pattern=parsed['FOR_QUERY'][key], repl=alt_parsed['NUMBER_LIKE'][key],
                                                 string=alt_address)
                            alt_parsed['TEXTUAL'] = strip_number_like(alt_address)

                            results.append(self._match_one_record(alt_parsed, row))

                # End of dealing with multiplier

                # Score and select

                try:
                    if self.scorer is not None:
                        X = [res['features'] for res in results]
#                         t0 = time.time() # time
                        scores = self.scorer.score_batch(X)
#                         self.t_log['scorer'] = time.time() - t0 # time

                        for res, score in zip(results, scores):
                            res['score'] = score

                            if table_name == 'Expanded':
                                res['score'] *= 0.9

                    best_res = max(results, key=lambda x: x['score'])
                    if best_res['score'] > score_threshold:
                        stop_loop = True

                except ValueError:
                    pass

#                 print(current_bw) # time

                if current_bw > max_beam_width:
                    stop_loop = True

                if stop_loop:
                    break

            if stop_loop:
                break

            # End of main loop on Tables

        # End the main loop

#         self._record_time('main_loop')  # time
#         self.t_log['beam_width'] = current_bw  # time

        if best_res is None:
            best_res = {'score': 0}

        if troubleshoot:
            return best_res, results
        else:
            return best_res

    def _match_safe(self, address):

        try:
            return self.match(address)

        except:
            try:
                return {'score': 0, 'error': '\n'.join([address, traceback.format_exc()])}
            except:
                return {'score': 0, 'error': traceback.format_exc()}

    def match_batch(self, list_of_addresses,
                    max_workers=None, chunksize=None):

        results = progress_map_tqdm_concurrent(self._match_safe, list_of_addresses,
                                               max_workers=max_workers, chunksize=chunksize)

        return results


class RuleScorer:

    """
    Rule scorer takes in a vector and return one scalar score
    """

    @staticmethod
    def weighted_score(x):
        x = np.array(x)
        w = np.array([5, 4, 3, 2, 1, 4, 5, 3, 2, 1, 3, 4, 5, 2, 1, 2, 3, 4, 5, 1, 1, 2, 3, 4, 5,
                      0, 0, -0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1])
        return sum(np.multiply(x, w))

    def __init__(self):
        pass

    def score(self, x, method='weighted_score'):
        score_func = self.__getattribute__(method)
        return score_func(x)

    def score_batch(self, X, method='weighted_score'):
        score_func = self.__getattribute__(method)
        return [score_func(x) for x in X]


class ClassifierScorer:

    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            estimator = pickle.load(f)

        estimator = estimator.set_params(verbose=0, n_jobs=1)

        self.estimator = estimator

    def score(self, x):
        score = self.estimator.predict_proba([x])[0, 1]
        return score

    def score_batch(self, X):
        score = list(self.estimator.predict_proba(X)[:, 1])
        return score


def query_task_to_sql(query_task, table_name):

    sql = f"""
    SELECT *
    FROM {table_name}
    WHERE %s
    """ % ("\nAND ".join(["%s = '%s'" % (k, v) for k, v in query_task.items()]))

    return sql


def repl_func(match):
    return '*' * len(match.group(0))


NUMBER_LIKE_MASTER_REGEX = re.compile(
    r"(FLAT|UNIT|HOUSE|BUILDING|ROOM|BLOCK|BONDS?|FL|APARTMENT|\(F|F|-|\()? ?"
    r"((\d[A-Z]\d+)|(PF|BF|GF|[A-Z])?(\d+)([A-Z])?|(?<!')(^|\b)([A-Z]|GROUND)($|\b)(?!'))"
    r"\)?")


def strip_number_like(address):
    p = NUMBER_LIKE_MASTER_REGEX

    p_pc = re.compile(r'([A-Z]{1,2})([0-9][A-Z0-9]?)(?: *\n?)?([0-9])([A-Z])([A-Z])')

    # postcode = re.search(p_pc, address).group()

    address = re.sub(p_pc, repl_func, address)

    address = re.sub(p, repl_func, address)
    #
    # address = re.sub('&&', postcode, address)

    return address


def prepare_uprn(row):

    p = re.compile(r"(FLAT|UNIT|BUILDING|ROOM|BLOCK|BONDS?|FL|APARTMENT|\()? ?"
                   r"((\d[A-Z]\d+)|([A-Z]|PF|BF|GF)?(\d+)([A-Z])?|(?<!')(^|\b)([A-Z]|GROUND)(\b|$)(?!'))")

    columns = ['ORGANISATION_NAME', 'DEPARTMENT_NAME', 'SUB_BUILDING_NAME', 'BUILDING_NAME', 'BUILDING_NUMBER',
               'DEPENDENT_THOROUGHFARE', 'THOROUGHFARE', 'DOUBLE_DEPENDENT_LOCALITY', 'DEPENDENT_LOCALITY',
               'POST_TOWN']

    d_text = {col: re.sub(p, repl_func, row[col]) for col in columns}

    number_like = [row[col] for col in ['number_like_0', 'number_like_1', 'number_like_2',
                                        'number_like_3', 'number_like_4']]

    postcode_split = {col: row[col] for col in ['pc_area', 'pc_district', 'pc_sector', 'pc_unit_0', 'pc_unit_1']}

    return {'TEXTUAL': d_text, 'NUMBER_LIKE': number_like, 'POSTCODE_SPLIT': postcode_split}


def get_number_like_matching_matrix(s1, s2):

    mat = [0] * 25

    for i in range(5):
        for j in range(i, 5):
            if all([s1[i] == '', s2[j] == '']):
                mat[i * 5 + j] = 0
            elif s1[i] == '':
                mat[i * 5 + j] = -1
            elif s2[j] == '':
                mat[i * 5 + j] = -2
            elif s1[i] == s2[j]:
                mat[i * 5 + j] = 1
            else:
                mat[i * 5 + j] = -3

    return mat


def postcode_matching(split1, split2):
    return [int(split1[k] == split2[k]) for k in split1]


def summarize_features(nl, ts, pm):

    return nl + list(ts.values()) + pm


def guess_multiplier(max_level, max_flat, n_tenement):

    if max_flat > 1:

        if max_level * max_flat == n_tenement:
            return max_flat, 'exact'

        elif n_tenement % max_flat == 0:
            return max_flat, 'high_for_flat'

        else:
            return max_flat, 'medium_for_flat'

    elif max_level > 1:

        if n_tenement % max_level == 0:
            return int(n_tenement / max_level), 'high_for_level'

    else:

        if (n_tenement % 2 == 0) and (n_tenement % 3 == 0):

            return 2, 'low'

        elif n_tenement % 2 == 0:

            return 2, 'low'

        elif n_tenement % 3 == 0:

            return 3, 'low'

        elif n_tenement % 5 == 0:
            return 5, 'low'

    return 2, 'guess'
