"""
Top level API for scoring existing matches

"""

import pandas as pd
import numpy as np
import csv
import os

import traceback

from flap.database.sql import SqlDB
from flap.matcher.sql_matcher import SqlMatcher
from flap.database.sql_in_memory import SqlDBInMemory
from flap.parser.rule_parser_fast import RuleParserFast
from flap.utils import join_uprn_fields, available_cpu_count


def csv_row_counter(filename):

    with open(filename, 'r') as f:
        row_count = sum(1 for _ in csv.reader(f))

    return row_count


def read_csv_header(filename):
    with open(filename, 'r') as f:
        dict_reader = csv.DictReader(f)
        headers = dict_reader.fieldnames

    return headers


def _load_all_csv_from_path(path, **kwargs):
    csv_files = [os.path.join(path, p) for p in os.listdir(path) if '.csv' in p]

    index_exists = _check_index(csv_files[0])

    if index_exists:
        dfs = [pd.read_csv(p, index_col=0, **kwargs) for p in csv_files]
    else:
        dfs = [pd.read_csv(p, **kwargs) for p in csv_files]

    return pd.concat(dfs)


def _check_index(csv_file):
    with open(csv_file, 'r') as f:
        header_0 = f.readline().split(',')[0]

    return header_0 == ''


def score(input_csv, db_path, output_file_path=None, raw_output_path=None,
          batch_size=10000, max_workers=None, in_memory_db=False, classifier_model_path=None,
          input_address_col='input_address', uprn_col='uprn'):

    # Initialise parameters

    if output_file_path is None:
        output_file_path = os.path.join(os.getcwd(), 'scoring_output.csv')

    if raw_output_path is None:
        raw_output_path = os.path.join(os.getcwd(), 'scoring_raw_output')

    if not os.path.exists(raw_output_path):
        os.mkdir(raw_output_path)

    if max_workers is None:
        max_workers = available_cpu_count()

    batch_size_adj = int(batch_size / max_workers) * max_workers
    total_tasks = csv_row_counter(input_csv) - 1
    total_batches = int(total_tasks / batch_size_adj) + int((total_tasks % batch_size_adj) > 0)

    # Check and read the input

    headers = read_csv_header(input_csv)
    assert all(s in headers for s in [input_address_col, uprn_col]), \
        f'Two columns are required in input csv file: `{input_address_col}` and `{uprn_col}`'
    batch_gen = pd.read_csv(input_csv, dtype='object', chunksize=batch_size_adj, index_col=0)

    # Check the database
    if not in_memory_db:
        sql_db = SqlDB(db_path)
        assert sql_db.db_status['table_indexed_built'], 'Database is not indexed, please Build the Database first'

        matcher = SqlMatcher(sql_db, scorer_path=classifier_model_path)
    else:
        sql_db = SqlDB(db_path)

        sql_db_in_memory = SqlDBInMemory()

        csv_files = [os.path.join(sql_db.sub_paths['csv'], file) for file in os.listdir(sql_db.sub_paths['csv'])]
        csv_names = [os.path.basename(file).split('.')[0]
                     for file in os.listdir(sql_db.sub_paths['csv'])]

        assert all([s in csv_names for s in ['indexed', 'expanded']]), 'CSV database does not exist'

        for table_name, path in zip(csv_names, csv_files):
            print(f'Loading Table {table_name} in memory')
            sql_db_in_memory.load_csv(path, table_name)
            print()

        parser = RuleParserFast(sql_db)

        matcher = SqlMatcher(sql_db_in_memory, parser=parser, scorer_path=classifier_model_path)

    # Main matching loop

    batch_index = 0

    while True:

        try:
            batch_name = 'batch_%s.csv' % batch_index
            print('Processing %s out of %s' % (batch_name, total_batches))

            batch_path = os.path.join(raw_output_path, batch_name)

            if not os.path.exists(batch_path):

                df_batch = next(batch_gen)

                mapper = {input_address_col: 'input_address', uprn_col: 'uprn'}
                rev_mapper = {'input_address': input_address_col, 'uprn': uprn_col}
                df_batch.rename(mapper, axis='columns')

                chunk_size = int(batch_size / max_workers) + 1

                res = matcher.score_matching_of_batch(df_batch, input_address_col=input_address_col, uprn_col=uprn_col,
                                                      max_workers=max_workers, chunksize=chunk_size)

                res.rename(rev_mapper, axis='columns')

                res.to_csv(batch_path)

            batch_index += 1

        except StopIteration:
            print('Scoring Finished, start summarising results')
            break

    results = _load_all_csv_from_path(raw_output_path)

    results.to_csv(output_file_path)

    print('Results can be see at: %s' % output_file_path)

    return results
