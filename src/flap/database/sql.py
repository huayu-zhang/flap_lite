import pandas as pd
import json
import os
import sqlite3
import shutil
from warnings import warn
from tqdm import tqdm

import flap

from flap.database.db_import import db_import
from flap.database.db_build_vocabulary import compile_pc1_uniques, compile_pc1_mappings_region, compile_global_uniques
from flap.database.db_index import db_index


MODULE_PATH = os.path.dirname(flap.__file__)


class SqlDBManager:

    def __init__(self, project_name=None):
        self.global_db_path = os.path.join(MODULE_PATH, 'db')
        self.project_name = project_name
        if project_name is not None:
            self.project_db_path = os.path.join(os.getcwd(), project_name, 'db')

    def list_global_db(self):
        db_names = os.listdir(self.global_db_path)
        return db_names

    def list_project_db(self):
        db_names = os.listdir(self.project_db_path)
        return db_names

    @staticmethod
    def create_db(path):
        new_db_path = path

        db_name = os.path.basename(path)

        if os.path.exists(new_db_path):
            warn('DB:%s exists, new database cannot be created.' % db_name)
        else:
            os.makedirs(new_db_path, exist_ok=True)
            print('New db created at: %s' % new_db_path)
            os.mkdir(os.path.join(new_db_path, 'raw'))
            os.mkdir(os.path.join(new_db_path, 'sql'))
            os.mkdir(os.path.join(new_db_path, 'db_config'))
            os.mkdir(os.path.join(new_db_path, 'csv'))

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
            os.mkdir(os.path.join(new_db_path, 'csv'))

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
            os.mkdir(os.path.join(new_db_path, 'csv'))

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

    def get_db(self, db_name, project_level=False):

        return SqlDB(self.get_db_path(db_name, project_level))


class SqlDB:

    def __init__(self, path_to_db):
        self.db_path = path_to_db
        self.db_name = os.path.basename(self.db_path)
        self.sub_paths = {
            'raw': os.path.join(self.db_path, 'raw'),
            'db_config': os.path.join(self.db_path, 'db_config'),
            'sql_db': os.path.join(self.db_path, 'sql', 'db.sqlite3'),
            'sql_db_temp': os.path.join(self.db_path, 'sql', 'db_temp.sqlite3'),
            'csv': os.path.join(self.db_path, 'csv'),
            'vocabulary': os.path.join(self.db_path, 'vocabulary'),
            'pc1_mappings': os.path.join(self.db_path, 'vocabulary', 'pc1_mappings.json'),
            'pc1_mappings_region': os.path.join(self.db_path, 'vocabulary', 'pc1_mappings_region.json'),
            'unique_DOUBLE_DEPENDENT_LOCALITY': os.path.join(self.db_path, 'vocabulary',
                                                             'unique_DOUBLE_DEPENDENT_LOCALITY.json'),
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

        try:
            cur.execute("""SELECT name FROM sqlite_schema""")
            res = cur.fetchall()

            if len(res):
                table_names = [list(name)[0] for name in res]
            else:
                table_names = []

        except sqlite3.OperationalError:
            table_names = []

        cur.close()
        conn.close()

        return table_names

    def get_columns_of_table(self, table_name):

        table_name = (table_name,)

        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute("""
        SELECT sql FROM sqlite_master WHERE tbl_name = ? AND type = 'table'
        """, table_name)
        res = cur.fetchall()

        try:
            s = res[0][0]
            lines = s.split('\n')
            column_names = [line.split('"')[1] for line in lines[1:-1]]
        except TypeError:
            column_names = []
        except IndexError:
            column_names = []

        cur.close()
        conn.close()

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
        db_status['table_indexed_built'] = 'indexed' in tables

        self.__db_status = db_status

        return self.__db_status

    def drop_table(self, table):
        conn = self.get_conn()
        cur = conn.cursor()

        try:
            cur.execute(f"""DROP TABLE {table}""")

        except sqlite3.OperationalError:
            print('Table not exist')

        conn.commit()
        cur.close()
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

    def setup_database(self, if_exists='skip'):
        self.build_raw(if_exists=if_exists)
        self.build_vocabulary()
        self.indexing_db(if_exists=if_exists)
        self.build_csv()
        print('Database Setup complete!')

    def build(self, if_exists='skip'):
        self.build_raw(if_exists=if_exists)
        self.build_vocabulary()
        self.indexing_db(if_exists=if_exists)
        self.build_csv()
        print('Database Setup complete!')
        
    def build_raw(self, if_exists='skip'):
        db_status = self.db_status

        print(db_status)

        if not db_status['raw_added']:
            raise FileNotFoundError('Add raw database files to [db_path]/raw first and then run build')

        if if_exists == 'skip':

            if db_status['table_raw_built']:
                print('Table raw exists, build skipped. Use if_exists="replace", if you want to create a new db')
            else:
                db_import(self)

        elif if_exists == 'replace':

            self.drop_table('raw')
            db_import(self)

    def indexing_db(self, if_exists='skip'):
        db_status = self.db_status

        if if_exists == 'skip':

            if db_status['table_indexed_built']:
                print('TABLE indexed exists, build skipped. Use if_exists="replace", if you want to create a new db')
            else:
                db_index(self)
                self.delete_temp()

        elif if_exists == 'replace':

            self.drop_table('indexed')
            db_index(self)
            self.delete_temp()

    def build_vocabulary(self):

        print('Start building vocabularies')
        if not os.path.exists(self.sub_paths['vocabulary']):
            os.mkdir(self.sub_paths['vocabulary'])

        compile_pc1_uniques(self)
        compile_pc1_mappings_region(self)
        compile_global_uniques(self)

        path_thoroughfare_patterns = os.path.join(MODULE_PATH, 'parser', 'thoroughfare_patterns.json')
        shutil.copy(path_thoroughfare_patterns, self.sub_paths['thoroughfare_patterns'])

        print('Finished building vocabularies')

    def build_csv(self):

        for table_name in ['indexed', 'expanded']:

            print(f'Start saving {table_name} to csv for in-memory DB')

            header = True

            cols = self.get_columns_of_table(table_name)

            sql = f'select * from {table_name}'

            chunk_gen = pd.read_sql(sql, con=self.get_conn(), columns=cols, chunksize=int(1e5))

            for chunk in tqdm(chunk_gen):

                chunk.to_csv(os.path.join(self.sub_paths['csv'], f'{table_name}.csv.gz'),
                             header=header, mode='a', compression='gzip')
                header = False

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

    def sql_query_by_column_values(self, table_name, column, value_list):

        in_clause = ', '.join(value_list)
        res = self.sql_query(f"""select * from {table_name} where {column} IN ({in_clause})""")
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

    def attach_uprn_fields_to_df(self, df, uprn_col='uprn'):

        uprn_list = [s for s in df[uprn_col].to_list() if isinstance(s, str)]

        res = self.sql_query_by_column_values(table_name='indexed', column='UPRN', value_list=uprn_list)
        columns = self.get_columns_of_table(table_name='indexed')
        res_df = pd.DataFrame.from_records(res, columns=columns)

        df = pd.merge(left=df, right=res_df, left_on=uprn_col, right_on='UPRN', how='left', suffixes=(None, '_db'))

        return df

    def create_index(self, table_name, columns):

        index_name = '__'.join(columns)

        sql = "CREATE INDEX IF NOT EXISTS %s ON %s(%s)" % (
            index_name,
            table_name,
            ', '.join(columns)
        )

        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql)
