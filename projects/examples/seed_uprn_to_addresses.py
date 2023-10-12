import os
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([os.getcwd()])


from src.database.sql import SqlDB
from src.matcher.sql_matcher import SqlMatcher
from src.parser.rule_parser_fast import RuleParserFast
from src.data.data import DataManager
from src.utils import join_uprn_fields

import pickle
from itertools import chain, islice
import pandas as pd


sql_db = SqlDB('/home/huayu_ssh/PycharmProjects/dres_r/db/scotland_curl')

matcher = SqlMatcher(sql_db)

csv_path = input('The path to csv: ')

output_path = input('The path for output: ')

input_df = pd.read_csv(csv_path, index_col=0)


def chunk_gen(iterable, size=5000):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


batch_index = 0

batch_size = 10000

number_of_batch = int(len(input_df) / batch_size) + 1

task_gen = chunk_gen(input_df.iterrows(), size=batch_size)


output_raw_path = output_path

while 1:
    try:

        batch_name = 'batch_%s.csv' % batch_index

        print(batch_name, number_of_batch)

        batch_path = os.path.join(output_raw_path, batch_name)

        if not os.path.exists(batch_path):

            batch = list(next(task_gen))

            address_list = [row['input_address'] for _, row in batch]

            results = matcher.match_batch(address_list, max_workers=12, chunksize=200)

            records = []

            for (_, row), result in zip(batch, results):

                record = {
                    'input_id': row['input_id'],
                    'input_address': row['input_address'],
                }

                try:
                    record['uprn_row'] = join_uprn_fields(result['uprn_row'])
                    record['uprn'] = result['uprn_row']['UPRN']
                    record['score'] = result['score']

                except KeyError:
                    try:
                        record['error'] = result['error']
                    except:
                        pass

                except TypeError:
                    pass

                records.append(record)

            records_df = pd.DataFrame.from_records(records)

            records_df.to_csv(batch_path)

        batch_index += 1

    except StopIteration:
        break
