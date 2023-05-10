# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:37:33 2022

@author: hzhang3410
"""

import os
import pickle
import pandas as pd
import textwrap
import json
import traceback
import warnings
import re

from uuid import uuid4
from time import ctime

from joblib import load

from src.parser.rule_parser_class import RuleParser
from src.database.sql import SqlDB
from src.preprocessing import postcode_preproc, address_line_preproc
from src.pipelines.nw_alignment import align_and_score
from src.alignment import SequentialAlignmentResults
from src.utils import repl_positions
from src.parser.multiplier_indexing import parse_multiplier


class RuleMatchByTree:

    def __init__(self, sql_db):
        self.sql_db = sql_db

        with open(sql_db.sub_paths['index_multiplier'], 'r') as f:
            self.indexed_multipliers = json.load(f)

        self.tree_path = sql_db.sub_paths['tree']

        self.log = None
        self.best_child = None
        self.last_children = None
        self.parsed = None

        self.rule_parser_class = RuleParser(sql_db=sql_db)
        self.rule_parser = self.rule_parser_class.parse

    def match(self, address, input_id=None, model=None, header=None):

        last_children = None
        best_child = None

        task_id = str(uuid4())

        try:
            address_p, preproc_log = address_line_preproc(address)

            parsed = self.rule_parser(address_p)

            if 'REGION' in parsed['SPAN']:
                start, end = parsed['SPAN']['REGION']
                address_p = repl_positions(address_p, list(range(start, end)))

            self.parsed = parsed

            address_rp = parsed['STRING']

            addresses_to_match = [address_p]

            if parsed['NUMBER_TYPE'] == 'MULTIPLIER':
                try:
                    key = '-'.join([address_rp['PRIMARY_NUMBER'], address_rp['POSTCODE']])
                    multiplier = self.indexed_multipliers[key]['multiplier']
                except NameError:
                    warnings.warn('Load multiplier index to name `indexed_multipliers` to use!')
                    multiplier = 2
                except KeyError:
                    multiplier = 2

                level, flat = parse_multiplier(address_rp['SECONDARY_NUMBER'])

                alt_sub_bnu = str((level - 1) * multiplier + flat)

                address_ex = re.sub(r'(\d[FKLPS]\d)', alt_sub_bnu, address_p)

                addresses_to_match.append(address_ex)

            last_children = []

            for address_candidate in addresses_to_match:

                for _dfs in _dfs_funcs:
                    p_node = _dfs(address_candidate, address_rp, self.tree_path)
                    if p_node is not None:
                        last_children.extend(_bfs(p_node))

            best_child = _select_best_child(last_children, model=model, header=header)

            if best_child is not None:
                log = {
                    'logging': {
                        'task_id': task_id,
                        'input_id': input_id,
                        'input_address': address,
                    },
                    'preproc': preproc_log,
                    'match': child_summary(best_child),
                    'db_record': best_child.metadata['Dataframe'].to_dict('records')[0]
                }
            else:
                log = {
                    'logging': {
                        'task_id': task_id,
                        'input_id': input_id,
                        'input_address': address
                    },
                    'preproc': preproc_log
                }

        except:
            log = {
                'logging': {
                    'task_id': task_id,
                    'input_id': input_id,
                    'input_address': address
                },
                'error': traceback.format_exc()
            }

        self.log = log
        self.last_children = last_children
        self.best_child = best_child

    def get_database_id(self):
        if self.log is None:
            self.match()

        if 'db_record' in self.log:
            return self.log['db_record']['UPRN']
        else:
            return None


def _dfs_by_postcode(address, address_rp, tree_path):

    if address is None:
        return None

    if 'POSTCODE' in address_rp:

        address_rp['POSTCODE'] = postcode_preproc(address_rp['POSTCODE'])

        pc0, pc1 = address_rp['POSTCODE'].split(' ')

        file = os.path.join(tree_path, 'pc_compiled', '-'.join([pc0, pc1]))

        try:
            with open(file, 'rb') as f:
                tree = pickle.load(f)

            ar = align_and_score(tree.name, address)
            ar.meta['name'] = tree.metadata['level']
            tree.metadata['align'] = SequentialAlignmentResults([ar])
            tree.metadata['end_address'] = repl_positions(address, [py for _, py in ar.meta['mapping']])

            return tree

        except FileNotFoundError:
            return None

    else:
        return None


def _dfs_by_pt_tho(address, address_rp, tree_path):

    if address is None:
        return None

    if ('POST_TOWN' in address_rp) and ('THOROUGHFARE' in address_rp):

        pt = address_rp['POST_TOWN']
        tho = address_rp['THOROUGHFARE']

        file = os.path.join(tree_path, 'pt_tho_compiled', '-'.join([pt, tho]))

        try:
            with open(file, 'rb') as f:
                tree = pickle.load(f)

            ar = align_and_score(tree.name, address)
            ar.meta['name'] = tree.metadata['level']
            tree.metadata['align'] = SequentialAlignmentResults([ar])
            tree.metadata['end_address'] = repl_positions(address, [py for _, py in ar.meta['mapping']])

            return tree

        except FileNotFoundError:

            if ('POST_TOWN' in address_rp) and ('STREET_1' in address_rp):

                pt = address_rp['POST_TOWN']
                tho = address_rp['STREET_1']

                file = os.path.join('./db/trees/pt_tho_compiled/', '-'.join([pt, tho]))

                try:
                    with open(file, 'rb') as f:
                        tree = pickle.load(f)

                    ar = align_and_score(tree.name, address)
                    ar.meta['name'] = tree.metadata['level']
                    tree.metadata['align'] = SequentialAlignmentResults([ar])
                    tree.metadata['end_address'] = repl_positions(address, [py for _, py in ar.meta['mapping']])

                    return tree

                except FileNotFoundError:
                    return None
    else:
        return None


_dfs_funcs = [_dfs_by_postcode, _dfs_by_pt_tho]


def _bfs(tree):

    children = tree.get_children()
    last_children = []

    while len(children):

        last_children = children.copy()

        children = []

        for child in last_children:

            text = child.name
            level = child.metadata['level']

            parent = child.get_parent()
            child.metadata['align'] = parent.metadata['align'].copy()

            if text == '':
                child.metadata['end_address'] = parent.metadata['end_address']

            else:
                last_address = parent.metadata['end_address']

                ar = align_and_score(child.name, last_address)
                ar.meta['name'] = level

                child.metadata['align'].append(ar)

                if level == 'POSTCODE':
                    match = re.search(r'([A-Z]{1,2}[0-9][A-Z0-9]?)(?: +)?([0-9][A-Z]{2})', last_address)
                    if match is not None:
                        start, end = match.span()
                        pys = list(range(start, end))
                        child.metadata['end_address'] = repl_positions(parent.metadata['end_address'], pys)
                    else:
                        child.metadata['end_address'] = parent.metadata['end_address']

                elif ar.meta['score'] > 0:

                    if ar.meta['perc_x'] > 0.3:
                        child.metadata['end_address'] = repl_positions(parent.metadata['end_address'],
                                                                       [py for _, py in ar.meta['mapping']])
                    else:
                        child.metadata['end_address'] = parent.metadata['end_address']

                else:
                    child.metadata['end_address'] = parent.metadata['end_address']

            children.extend(child.get_children())

    return last_children


def _select_best_child(last_children, model=None, header=None):

    if model is None:

        for child in last_children:
            child.metadata['align'].score_alignments_by_rule()

        best_child = max(last_children, key=lambda child: child.metadata['align'].aggregated_alignment_score['total_score'])

    else:

        l_feature = [child.metadata['align'].get_alignment_features() for child in last_children]

        X = pd.DataFrame.from_records(l_feature).to_numpy()

        y_prob = model.predict_proba(X)[:, 1]

        best_child_index = np.argmax(y_prob)

        best_child = last_children[best_child_index]

    return best_child


def child_summary(child):
    res = {
        'matched': str(child.metadata['Dataframe']['UPRN'].item()),
        'alignment_view': str(child.metadata['align']),
        'alignment_scores': child.metadata['align'].aggregated_alignment_score,
        'db_record': child.metadata['Dataframe'].to_dict('records')[0]
    }
    return res


def generate_features(task_id, input_id, child):

    res = {
        'task_id': task_id,
        'input_id': input_id,
        'db_id': str(child.metadata['Dataframe']['UPRN'].item())
    }
    res.update(child.metadata['align'].get_alignment_features())

    return res


def _run_task(args):
    address, input_id, output_path = args

    output_file = os.path.join(output_path, '%s.json' % input_id)

    if not os.path.exists(output_file):
        matcher = RuleMatchByTree(address)
        matcher.match(input_id)

        log = matcher.log

        with open(output_file, 'w') as f:
            json.dump(log, f, indent=4)


def run(input_path, output_path, subset=None, parallel=True):

    input_df = pd.read_csv(input_path)

    if subset is not None:
        input_df = input_df.loc[input_df['input_id'].isin(subset)]

    task_generator = ((row['input_address'], row['input_id'], output_path) for _, row in input_df.iterrows())

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if parallel:
        from tqdm.contrib.concurrent import process_map
        process_map(_run_task, task_generator)
    else:
        try:
            from tqdm import tqdm
            for args in tqdm(task_generator):
                _run_task(args)
        except ModuleNotFoundError:
            for args in task_generator:
                _run_task(args)

#
# sql_db = SqlDB('/home/huayu_ssh/PycharmProjects/dres_r/db/scotland_20200910')
#
# rm = RuleMatchByTree('27/25 BRUNSWICK ROAD EDINBURGH EH7 5GX', sql_db=sql_db)
#
# rm.match()
#
# rm.log