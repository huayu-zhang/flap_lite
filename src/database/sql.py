from zipfile import ZipFile
import csv
import io
import pandas as pd
import geopandas as gpd
import json
import os
import tqdm
import sqlite3
import re
import itertools
import pickle

from tqdm.contrib.concurrent import process_map
from src.database_compile.db_expansion import expand_uprn
from warnings import warn

from src.tree import Tree


class SqlDBManager:

    def __init__(self, project_name=None):
        self.global_db_path = os.path.join(os.getcwd(), 'db')
        self.project_name = project_name
        if project_name is not None:
            self.project_db_path = os.path.join(os.getcwd(), project_name, 'db')

    def list_global_db(self):
        db_names = os.listdir(self.global_db_path)
        print(db_names)
        return db_names

    def list_project_db(self):
        db_names = os.listdir(self.project_db_path)
        print(db_names)
        return db_names

    def create_global_db(self, db_name):
        new_db_path = os.path.join(self.global_db_path, db_name)

        if os.path.exists(new_db_path):
            warn('DB:%s exists, new database cannot be created.' % db_name)
        else:
            os.makedirs(new_db_path, exist_ok=True)
            print('New db created at: %s' % new_db_path)
            os.mkdir(os.path.join(new_db_path, 'raw'))
            os.mkdir(os.path.join(new_db_path, 'sql'))
            os.mkdir(os.path.join(new_db_path, 'db_config'))

    def create_project_db(self, db_name):
        new_db_path = os.path.join(self.project_db_path, db_name)

        if os.path.exists(new_db_path):
            warn('DB:%s exists, new database cannot be created.' % db_name)
        else:
            os.makedirs(new_db_path, exist_ok=True)
            print('New db created at: %s' % new_db_path)
            os.mkdir(os.path.join(new_db_path, 'raw'))
            os.mkdir(os.path.join(new_db_path, 'sql'))
            os.mkdir(os.path.join(new_db_path, 'db_config'))

    def get_db_path(self, db_name, project_level=False):
        if not project_level:
            if db_name in self.list_global_db():
                return os.path.join(self.global_db_path, db_name)
            else:
                warn('DB: %s does not exist!' % db_name)
        else:
            if self.project_name is None:
                warn('Project not specified')
            else:
                if db_name in self.list_project_db():
                    return os.path.join(self.project_db_path, db_name)
                else:
                    warn('DB: %s does not exist!' % db_name)


class SqlDB:

    def __init__(self, path_to_db):
        self.db_path = path_to_db
        self.db_name = os.path.basename(self.db_path)
        self.sub_paths = {
            'raw': os.path.join(self.db_path, 'raw'),
            'db_config': os.path.join(self.db_path, 'db_config'),
            'sql_db': os.path.join(self.db_path, 'sql', 'db.sqlite3'),
            'sql_db_temp': os.path.join(self.db_path, 'sql', 'db_temp.sqlite3'),
            'tree': os.path.join(self.db_path, 'tree'),
            'index': os.path.join(self.db_path, 'index'),
            'index_database_local': os.path.join(self.db_path, 'index', 'index_database_local.json'),
            'index_multiplier': os.path.join(self.db_path, 'index', 'index_multiplier.json'),
            'vocabulary': os.path.join(self.db_path, 'vocabulary'),
            'pc1_mappings': os.path.join(self.db_path, 'vocabulary', 'pc1_mappings.json'),
            'pc1_mappings_region': os.path.join(self.db_path, 'vocabulary', 'pc1_mappings_region.json'),
            'unique_DOUBLE_DEPENDENT_LOCALITY': os.path.join(self.db_path, 'vocabulary', 'unique_DOUBLE_DEPENDENT_LOCALITY.json'),
            'unique_DEPENDENT_LOCALITY': os.path.join(self.db_path, 'vocabulary', 'unique_DEPENDENT_LOCALITY.json'),
            'unique_POST_TOWN': os.path.join(self.db_path, 'vocabulary', 'unique_POST_TOWN.json'),
            'thoroughfare_patterns': os.path.join(self.db_path, 'vocabulary', 'thoroughfare_patterns.json')
        }

        self.__db_status = None

    def __str__(self):
        self_str = f"""
        Database name: {self.db_name}
        Database path: {self.db_path}
        """
        return self_str

    def get_table_names(self):
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute("""SELECT name FROM sqlite_schema""")
        res = cur.fetchall()

        if len(res):
            table_names = [list(name)[0] for name in res]
        else:
            table_names = []

        cur.close()
        conn.close()
        return table_names

    def get_columns_of_table(self, table_name):

        table_name = (table_name, )

        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute("""
        SELECT sql FROM sqlite_master WHERE tbl_name = ? AND type = 'table'
        """, table_name)
        res = cur.fetchall()

        s = res[0][0]
        lines = s.split('\n')
        column_names = [line.split('"')[1] for line in lines[1:-1]]

        return column_names

    @property
    def db_status(self):
        db_status = {
            'db_name': self.db_name,
            'db_path': self.db_path,
            'valid': os.path.isdir(self.db_path),
            'raw_added': len(os.listdir(self.sub_paths['raw'])) > 0
        }

        tables = self.get_table_names()

        db_status['table_raw_built'] = 'raw' in tables
        db_status['table_ex_built'] = 'ex' in tables

        self.__db_status = db_status

        return self.__db_status

    def drop_table(self, table):
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(f"""DROP TABLE IF EXISTS {table}""")
        conn.commit()
        conn.close()

    def get_db_config(self):

        sql_config_path = [os.path.join(self.sub_paths['db_config'], file)
                           for file in os.listdir(self.sub_paths['db_config']) if '.json' in file]

        if len(sql_config_path):

            sql_config_file = sql_config_path[0]

            with open(sql_config_file, 'r') as f:
                sql_config = json.load(f)

        else:

            sql_config = {
                "RECORD_IDENTIFIER": "INTEGER",
                "CHANGE_TYPE": "TEXT",
                "PRO_ORDER": "TEXT",
                "UPRN": "TEXT",
                "UDPRN": "TEXT",
                "ORGANISATION_NAME": "TEXT",
                "DEPARTMENT_NAME": "TEXT",
                "SUB_BUILDING_NAME": "TEXT",
                "BUILDING_NAME": "TEXT",
                "BUILDING_NUMBER": "TEXT",
                "DEPENDENT_THOROUGHFARE": "TEXT",
                "THOROUGHFARE": "TEXT",
                "DOUBLE_DEPENDENT_LOCALITY": "TEXT",
                "DEPENDENT_LOCALITY": "TEXT",
                "POST_TOWN": "TEXT",
                "POSTCODE": "TEXT",
                "POSTCODE_TYPE": "TEXT",
                "DELIVERY_POINT_SUFFIX": "TEXT",
                "WELSH_DEPENDENT_THOROUGHFARE": "TEXT",
                "WELSH_THOROUGHFARE": "TEXT",
                "WELSH_DOUBLE_DEPENDENT_LOCALITY": "TEXT",
                "WELSH_DEPENDENT_LOCALITY": "TEXT",
                "WELSH_POST_TOWN": "TEXT",
                "PO_BOX_NUMBER": "TEXT",
                "PROCESS_DATE": "TEXT",
                "START_DATE": "TEXT",
                "END_DATE": "TEXT",
                "LAST_UPDATE_DATE": "TEXT",
                "ENTRY_DATE": "TEXT"
            }

            sql_config_file = os.path.join(self.sub_paths['db_config'], 'db_config.json')

            with open(sql_config_file, 'w') as f:
                json.dump(sql_config, f, indent=4)

            print('db_config created')
        return sql_config

    def build_raw(self, if_exist='skip'):
        db_status = self.db_status

        print(db_status)

        if not db_status['raw_added']:
            raise FileNotFoundError('Add raw database files to [db_path]/raw first and then run build')

        if if_exist == 'skip':

            if db_status['table_raw_built']:
                print('Table raw exists, build skipped. Use if_exist="replace", if you want to create a new db')
            else:
                _parsing_func(self)

        elif if_exist == 'replace':

            self.drop_table('raw')
            _parsing_func(self)

    def build_ex(self, if_exist='skip'):
        db_status = self.db_status

        if if_exist == 'skip':

            if db_status['table_ex_built']:
                print('TABLE ex exists, build skipped. Use if_exist="replace", if you want to create a new db')
            else:
                _ex_func(self)
                self.delete_temp()

        elif if_exist == 'replace':

            self.drop_table('ex')
            _ex_func(self)
            self.delete_temp()

    def build_trees(self):
        compile_postcode_trees(self)
        compile_pt_tho_trees(self)

    def build_vocabulary(self):

        if not os.path.exists(self.sub_paths['vocabulary']):
            os.mkdir(self.sub_paths['vocabulary'])

        compile_pc1_uniques(self)
        compile_pc1_mappings_region(self)
        compile_global_uniques(self)

        path_thoroughfare_patterns = os.path.join(os.getcwd(), 'src', 'parser', 'thoroughfare_patterns.json')
        os.system('cp %s %s' % (path_thoroughfare_patterns, self.sub_paths['thoroughfare_patterns']))

    def build_indices(self):

        if not os.path.exists(self.sub_paths['index']):
            os.mkdir(self.sub_paths['index'])

        build_index_for_database(self)

    def setup_database(self):
        self.build_raw()
        self.build_ex()
        self.build_trees()
        self.build_vocabulary()
        self.build_indices()

    def reset_database(self):
        print('Deleting SQL')
        os.system('rm %s' % self.sub_paths['sql_db'])

        print('Deleting Trees')
        os.system('rm -rf %s/*' % self.sub_paths['tree'])

    def get_conn(self):
        return sqlite3.connect(self.sub_paths['sql_db'], timeout=10)

    def get_conn_temp(self):
        return sqlite3.connect(self.sub_paths['sql_db_temp'], timeout=10)

    def delete_temp(self):
        try:
            os.remove(self.sub_paths['sql_db_temp'])
        except FileNotFoundError:
            pass

    def sql_query(self, query):

        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(query)
        res = cur.fetchall()
        cur.close()
        conn.close()

        return res

    def sql_query_batch(self, query, batch_size=int(1e5)):
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(query)

        for batch in iter(lambda: cur.fetchmany(batch_size), []):
            yield batch

        conn.close()

    def sql_table_batch_by_column(self, table_name, by_columns, batch_size, match_func=None):

        if isinstance(by_columns, str):
            by_columns = [by_columns]

        table_columns = self.get_columns_of_table(table_name)

        column_indices = [table_columns.index(col) for col in by_columns]

        query = """
            SELECT *
            FROM %s
            ORDER BY %s
        """ % (table_name, ','.join(by_columns))

        con = self.get_conn()
        cur = con.cursor()
        cur.execute(query)

        res = cur.fetchmany(batch_size)
        next_res = cur.fetchone()

        while len(res):

            try:

                if match_func is None:
                    while all([next_res[i] == res[-1][i] for i in column_indices]):
                        res.append(next_res)
                        next_res = cur.fetchone()
                else:
                    while match_func(next_res, res):
                        res.append(next_res)
                        next_res = cur.fetchone()

            except TypeError:
                pass

            res_df = pd.DataFrame.from_records(res)
            res_df.columns = table_columns

            yield res_df

            if next_res is None:
                break
            else:
                res = [next_res]
                res.extend(cur.fetchmany(batch_size))
                next_res = cur.fetchone()

        con.close()


def numeric_to_object(x):

    if pd.isna(x):
        return ''

    elif isinstance(x, int):
        return str(x)

    elif isinstance(x, float):
        return str(int(x))

    else:
        return x


def convert_df_to_object(df):

    for col in df:

        if pd.api.types.is_object_dtype(df[col]):
            df.loc[:, col] = df[col].fillna('')

        elif pd.api.types.is_numeric_dtype(df[col]):
            df.loc[:, col] =  df[col].apply(lambda x: numeric_to_object(x))

        else:
            df.loc[:, col] =  df[col]

    return df


def _parsing_func(sql_db):

    zip_path = [os.path.join(sql_db.sub_paths['raw'], file)
                for file in os.listdir(sql_db.sub_paths['raw']) if '.zip' in file][0]

    db_file = sql_db.sub_paths['sql_db']

    conn = sqlite3.connect(db_file)

    with ZipFile(zip_path, 'r') as z:
        # Get file names of the big zip file
        zz_files = [zz_file for zz_file in z.namelist()]

        if any('.gpkg' in zz_file for zz_file in zz_files):

            print('.gpkg database found. Starting to read db file')

            zz_data = io.BytesIO(z.read(zz_files[0]))

            gpkg_db = gpd.read_file(zz_data, layer='delivery_point_address')

            print('convert to object type')

            gpkg_db = convert_df_to_object(gpkg_db)

            cols_upper = [col.upper() for col in gpkg_db.columns]

            gpkg_db.columns = cols_upper

            print('Save to SQL database')

            gpkg_db.to_sql(name='raw', con=conn, if_exists='replace', dtype='TEXT', index=False)

            print('Raw table building finished')

        elif any('.zip' in zz_file for zz_file in zz_files):
            print('.zip database found. Starting to read db file')

            chunk_size = int(1e5)
            rows = []
            data_index = '28'

            sql_config = sql_db.get_db_config()

            zz_files = [zz_file for zz_file in z.namelist() if '.zip' in zz_file]

            for zz_file in tqdm.tqdm(zz_files):  # Load data of all sub zips and get the csv file
                zz_data = io.BytesIO(z.read(zz_file))

                with ZipFile(zz_data, 'r') as zz:
                    csv_files = [csv_file for csv_file in zz.namelist() if '.csv' in csv_file]

                    for csv_file in csv_files:

                        with zz.open(csv_file, 'r') as f:

                            lines = [line.decode() for line in f.readlines()]

                            reader = csv.reader(lines)

                            for row in reader:
                                if row[0] == data_index:
                                    rows.append(row)

                    while len(rows) > chunk_size:
                        df = pd.DataFrame.from_records(rows, columns=list(sql_config.keys()))
                        df.to_sql(name='raw', con=conn, if_exists='append', dtype=sql_config, index=False)
                        rows = rows[chunk_size:]

        else:
            pass

    conn.close()


def _ex_func(sql_db):

    conn_temp = sql_db.get_conn_temp()

    tasks_gen = sql_db.sql_table_batch_by_column('raw', 'POSTCODE', int(1e5))

    sql_config = sql_db.get_db_config()

    sql_ex_config = sql_config.copy()

    sql_ex_config.update({
        "ex_label": "TEXT",
        "ex_uprn": "TEXT"}
    )

    for res in tqdm.tqdm(tasks_gen):
        res_ex = expand_uprn(res)
        res_ex.to_sql(name='ex',
                      con=conn_temp,
                      if_exists='append',
                      dtype=sql_ex_config,
                      index=False)

    sql_chunk_gen = pd.read_sql(sql='SELECT * FROM ex', con=conn_temp, chunksize=int(1e5))

    conn = sql_db.get_conn()

    for chunk in tqdm.tqdm(sql_chunk_gen):
        chunk.to_sql(name='ex',
                     con=conn,
                     if_exists='append',
                     dtype=sql_ex_config,
                     index=False
                     )

    conn.close()
    conn_temp.close()


seq_levels = {
    'pc': [
            'POST_TOWN', 'THOROUGHFARE',
            'DEPENDENT_LOCALITY', 'DOUBLE_DEPENDENT_LOCALITY',
             'DEPENDENT_THOROUGHFARE',
            'BUILDING_NAME', 'BUILDING_NUMBER', 'SUB_BUILDING_NAME',
            'DEPARTMENT_NAME', 'ORGANISATION_NAME'
            ],


    'pt_tho': [
            'POSTCODE',
            'DEPENDENT_LOCALITY', 'DOUBLE_DEPENDENT_LOCALITY',
            'DEPENDENT_THOROUGHFARE',
            'BUILDING_NAME', 'BUILDING_NUMBER', 'SUB_BUILDING_NAME',
            'DEPARTMENT_NAME', 'ORGANISATION_NAME'
            ]
        }


def add_and_return_children(args):

    level, node = args

    children = [Tree(name, None, {'level': level, 'Dataframe': group})
                for name, group in node.metadata['Dataframe'].groupby(level)]
    node.add_children(children)
    node.metadata['Dataframe'] = None

    return children


def df_to_tree(df, seq_levels, verbose=True):

    tree = Tree('root', None, {'level': 'root', 'Dataframe': df})
    current_nodes = [tree]

    for level in seq_levels:

        if verbose:
            next_nodes = list(
                itertools.chain(
                    *progress_map(add_and_return_children,
                                  list(map(lambda node: (level, node), current_nodes)))
                )
            )
        else:
            next_nodes = list(
                itertools.chain(
                    *map(add_and_return_children,
                         list(map(lambda node: (level, node), current_nodes)))
                )
            )
        current_nodes = next_nodes

    for node in current_nodes:
        if len(node.metadata['Dataframe']) > 1:

            warn('Potential duplicated records from expansion: \n %s'
                          % str(node.metadata['Dataframe'].to_dict('records')))

            original = ['-' not in str(index) for index in node.metadata['Dataframe'].index]

            if sum(original):
                node.metadata['Dataframe'] = node.metadata['Dataframe'][original]
            else:
                node.metadata['Dataframe'] = node.metadata['Dataframe'].iloc[0, :]

    return tree


def _compile_postcode_subtree(args):

    path, postcode, group = args

    pc0, pc1 = postcode.split(' ')

    file = os.path.join(path, '-'.join([pc0, pc1]))

    if not os.path.exists(file):
        sub_tree = df_to_tree(group, seq_levels['pc'], verbose=False)

        sub_tree.name = postcode
        sub_tree.metadata['level'] = 'POSTCODE'

        with open(file, 'wb') as f:
            pickle.dump(sub_tree, f)


def compile_postcode_trees(sql_db, parallel=True):

    df_batch = sql_db.sql_table_batch_by_column(table_name='ex', by_columns='POSTCODE', batch_size=int(1e5))

    path = os.path.join(sql_db.sub_paths['tree'], 'pc_compiled')

    if not os.path.exists(path):
        os.makedirs(path)

    try:
        while 1:
            df = next(df_batch)

            if parallel:
                from tqdm.contrib.concurrent import process_map

                process_map(_compile_postcode_subtree, ((path, name, group) for name, group in df.groupby('POSTCODE')),
                            max_workers=os.cpu_count())
            else:
                for name, group in df.groupby('POSTCODE'):
                    _compile_postcode_subtree((name, group))

    except StopIteration:
        pass

    finally:
        del df_batch


def _compile_pt_tho_subtree(args):

    path, (pt, tho), group = args

    file = os.path.join(path, '-'.join([pt, tho]))

    if not os.path.exists(file):
        sub_tree = df_to_tree(group, seq_levels['pt_tho'], verbose=False)

        sub_tree.name = tho
        sub_tree.metadata['level'] = 'THOROUGHFARE'

        parent_tree = Tree(pt, sub_tree, {'level': 'POST_TOWN'})

        with open(file, 'wb') as f:
            pickle.dump(parent_tree, f)


def compile_pt_tho_trees(sql_db, parallel=True):

    df_batch = sql_db.sql_table_batch_by_column(table_name='ex', by_columns=['POST_TOWN', 'THOROUGHFARE'],
                                                batch_size=int(1e5))

    path = os.path.join(sql_db.sub_paths['tree'], 'pt_tho_compiled')

    if not os.path.exists(path):
        os.makedirs(path)

    try:
        while 1:
            df = next(df_batch)

            if parallel:
                from tqdm.contrib.concurrent import process_map

                process_map(_compile_pt_tho_subtree, ((path, name, group) for name, group in df.groupby(['POST_TOWN', 'THOROUGHFARE'])),
                            max_workers=os.cpu_count())
            else:
                for name, group in df.groupby(['POST_TOWN', 'THOROUGHFARE']):
                    _compile_pt_tho_subtree((name, group))

    except StopIteration:
        pass

    finally:
        del df_batch


def compile_pc1_uniques(sql_db):

    batch_gen = sql_db.sql_table_batch_by_column(
        table_name='ex', by_columns='POSTCODE', batch_size=int(1e5),
        match_func=lambda next_res, res: next_res[15].split(' ')[0] == res[-1][15].split(' ')[0])

    columns_to_compile = ["THOROUGHFARE", "DOUBLE_DEPENDENT_LOCALITY", "DEPENDENT_LOCALITY", "POST_TOWN"]

    pc1_mappings = {}

    for col in columns_to_compile:
        pc1_mappings[col] = {}

    for df in tqdm.tqdm(batch_gen):

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

    path_pc1_region_mappings = os.path.join(os.getcwd(), 'src', 'database', 'pc1_mappings_region.json')
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


def index_local_neighbourhood_database(batch):

    collections = []

    for (bnu, pc, ex_label), group in batch.groupby(['BUILDING_NUMBER', 'POSTCODE', 'ex_label']):

        if 'syn' not in ex_label:
            if len(bnu):
                if len(group) > 1:
                    collections.append({
                        'index': '-'.join([bnu, pc]),
                        'number_of_subs': len(group)
                    })
    return collections


def guess_multiplier(index, local_max, local_n_subs):

    if (index in local_max) and (index in local_n_subs):

        max_level = local_max[index]['max_level']
        max_flat = local_max[index]['max_flat']
        n_sub = local_n_subs[index]

        if max_flat > 1:

            if max_level * max_flat == n_sub:
                return {'multiplier': max_flat, 'confidence': 'exact'}

            elif n_sub % max_flat == 0:
                return {'multiplier': max_flat, 'confidence': 'high'}

    elif index in local_n_subs:

        n_sub = local_n_subs[index]

        if (n_sub % 2 == 0) and (n_sub % 3 == 0):

            return {'multiplier': 2, 'confidence': 'low'}

        elif n_sub % 2 == 0:

            return {'multiplier': 2, 'confidence': 'medium'}

        elif n_sub % 3 == 0:

            return {'multiplier': 3, 'confidence': 'medium'}

    elif index in local_max:

        max_flat = local_max[index]['max_flat']

        if max_flat > 1:

            return {'multiplier': max_flat, 'confidence': 'low'}

    return {'multiplier': 2, 'confidence': 'none'}


def build_index_for_database(sql_db):

    batch_gen = sql_db.sql_table_batch_by_column(table_name='ex', by_columns='POSTCODE', batch_size=int(1e5))

    list_local_index_database = process_map(index_local_neighbourhood_database, batch_gen)

    list_local_index_database_flattened = list(itertools.chain(*list_local_index_database))

    local_index_database = {d['index']: d['number_of_subs'] for d in list_local_index_database_flattened}

    index_multiplier = {}

    for k, v in local_index_database.items():

        n_sub = v

        if n_sub % 16 == 0:
            index_multiplier[k] = {'multiplier': 2, 'confidence': 'medium'}

        elif n_sub % 12 == 0:
            index_multiplier[k] = {'multiplier': 3, 'confidence': 'low'}

        elif (n_sub % 2 == 0) and (n_sub % 3 == 0):
            index_multiplier[k] = {'multiplier': 2, 'confidence': 'low'}

        elif n_sub % 2 == 0:
            index_multiplier[k] = {'multiplier': 2, 'confidence': 'medium'}

        elif n_sub % 3 == 0:
            index_multiplier[k] = {'multiplier': 3, 'confidence': 'medium'}

        else:
            index_multiplier[k] = {'multiplier': 2, 'confidence': 'low'}

    with open(sql_db.sub_paths['index_database_local'], 'w') as f:
        json.dump(local_index_database, f, indent=4)

    with open(sql_db.sub_paths['index_multiplier'], 'w') as f:
        json.dump(index_multiplier, f, indent=4)



# sql_db = SqlDB('/home/huayu_ssh/PycharmProjects/dres_r/db/scotland_20200910')
#
# sql_db.build_trees()
