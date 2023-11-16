"""
Process of DB build:

Read raw data

Create Range expansion

Index raw data and range expansion

Expansion

"""

import os
import json
from tqdm import tqdm
import itertools
import flap


MODULE_PATH = os.path.dirname(flap.__file__)


def compile_pc1_uniques(sql_db):
    batch_gen = sql_db.sql_table_batch_by_column(
        table_name='raw', by_columns='POSTCODE', batch_size=int(1e5),
        match_func=lambda next_res, res: next_res[15].split(' ')[0] == res[-1][15].split(' ')[0])

    columns_to_compile = ["THOROUGHFARE", "DOUBLE_DEPENDENT_LOCALITY", "DEPENDENT_LOCALITY", "POST_TOWN"]

    pc1_mappings = {}

    for col in columns_to_compile:
        pc1_mappings[col] = {}

    for df in tqdm(batch_gen):

        df['pc1'] = df.apply(lambda row: row['POSTCODE'].split(' ')[0], axis=1)

        for pc1, group in df.groupby('pc1'):

            for col in columns_to_compile:

                uniques = group[col].unique().tolist()

                try:
                    uniques.remove('')
                except ValueError:
                    pass

                if len(uniques):
                    pc1_mappings[col][pc1] = uniques

    with open(sql_db.sub_paths['pc1_mappings'], 'w') as f:
        json.dump(pc1_mappings, f, indent=4)


def compile_pc1_mappings_region(sql_db):
    path_pc1_region_mappings = os.path.join(MODULE_PATH, 'parser', 'pc1_mappings_region.json')
    os.system('cp %s %s' % (path_pc1_region_mappings, sql_db.sub_paths['pc1_mappings_region']))


def compile_global_uniques(sql_db):
    columns_to_compile = ["DOUBLE_DEPENDENT_LOCALITY", "DEPENDENT_LOCALITY", "POST_TOWN"]

    for col in columns_to_compile:
        res_sql = sql_db.sql_query("""
                                   SELECT DISTINCT %s
                                   FROM raw
                                   """ % col)
        res = list(itertools.chain(*res_sql))

        try:
            res.remove('')
        except ValueError:
            pass

        with open(sql_db.sub_paths['unique_%s' % col], 'w') as f:
            json.dump(res, f, indent=4)
