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

from src.utils import flatten


class SqlDBManager:

    def __init__(self, project_name=None):
        self.global_db_path = os.path.join(os.getcwd(), 'db')
        self.project_name = project_name
        if project_name is not None:
            self.project_db_path = os.path.join(os.getcwd(), project_name, 'db')

    def list_global_db(self):
        db_names = os.listdir(self.global_db_path)
        return db_names

    def list_project_db(self):
        db_names = os.listdir(self.project_db_path)
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

        table_name = (table_name,)

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
        db_status['table_indexed_built'] = 'indexed' in tables
        

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

    def indexing_db(self, if_exist='skip'):
        db_status = self.db_status

        if if_exist == 'skip':

            if db_status['table_indexed_built']:
                print('TABLE indexed exists, build skipped. Use if_exist="replace", if you want to create a new db')
            else:
                _indexing_database(self)
                self.delete_temp()

        elif if_exist == 'replace':

            self.drop_table('indexed')
            _indexing_database(self)
            self.delete_temp()

    def build_vocabulary(self):

        if not os.path.exists(self.sub_paths['vocabulary']):
            os.mkdir(self.sub_paths['vocabulary'])

        compile_pc1_uniques(self)
        compile_pc1_mappings_region(self)
        compile_global_uniques(self)

        path_thoroughfare_patterns = os.path.join(os.getcwd(), 'src', 'parser', 'thoroughfare_patterns.json')
        os.system('cp %s %s' % (path_thoroughfare_patterns, self.sub_paths['thoroughfare_patterns']))

    def setup_database(self, if_exist='skip'):
        self.build_raw(if_exist=if_exist)
        self.build_vocabulary()
        self.indexing_db(if_exist=if_exist)

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


def _parsing_func(sql_db):
    zip_path = [os.path.join(sql_db.sub_paths['raw'], file)
                for file in os.listdir(sql_db.sub_paths['raw']) if '.zip' in file][0]

    db_file = sql_db.sub_paths['sql_db']

    conn = sqlite3.connect(db_file)

    with ZipFile(zip_path, 'r') as z:
        
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

            for zz_file in tqdm.tqdm(zz_files):  
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

            df = pd.DataFrame.from_records(rows, columns=list(sql_config.keys()))
            df.to_sql(name='raw', con=conn, if_exists='append', dtype=sql_config, index=False)
            print('Raw table building finished')

        else:
            pass

    conn.close()


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
    path_pc1_region_mappings = os.path.join(os.getcwd(), 'src', 'parser', 'pc1_mappings_region.json')
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


def _indexing_database(sql_db):
    conn_temp = sql_db.get_conn_temp()

    tasks_gen = sql_db.sql_table_batch_by_column('raw', 'POSTCODE', int(1e5))

    for res in tqdm.tqdm(tasks_gen):
        res_indexed = indexing_uprn(res)
        res_indexed.to_sql(name='indexed',
                           con=conn_temp,
                           if_exists='append',
                           dtype='TEXT',
                           index=False)

    sql_chunk_gen = pd.read_sql(sql='SELECT * FROM indexed', con=conn_temp, chunksize=int(1e5))

    conn = sql_db.get_conn()

    for chunk in tqdm.tqdm(sql_chunk_gen):
        chunk.to_sql(name='indexed',
                     con=conn,
                     if_exists='append',
                     dtype='TEXT',
                     index=False
                     )

    conn.close()
    conn_temp.close()


def type_of_macro(row):

    cols = ['DEPENDENT_THOROUGHFARE', 'THOROUGHFARE', 'DOUBLE_DEPENDENT_LOCALITY', 'DEPENDENT_LOCALITY']

    return ''.join([str(int(len(row[col]) > 0)) for col in cols])


def type_of_micro(row):

    cols = ['ORGANISATION_NAME', 'DEPARTMENT_NAME', 'SUB_BUILDING_NAME',
            'BUILDING_NAME', 'BUILDING_NUMBER']

    return ''.join([str(int(len(row[col]) > 0)) for col in cols])


def parse_number_like_uprn(row):

    p = re.compile(r"((\d[A-Z]\d+)|([A-Z]|PF|BF|GF)?(\d+)([A-Z])?|(?<!')(^|\b)([A-Z]|GROUND)(\b|$)(?!'))")

    split_pattern = re.compile(r"(?<=\d)(?=\D)|(?=\d)(?<=\D)")

    units = list()

    for col in ['SUB_BUILDING_NAME', 'BUILDING_NAME']:
        if len(row[col]):
            matches = list(re.finditer(p, row[col]))
            units.append(flatten([re.split(split_pattern, match.group(1))
                          if re.search(r'\dF\d', match.group(1)) is None else [match.group(1)]
                          for match in matches]))

    if len(row['BUILDING_NUMBER']):
        units.append([row['BUILDING_NUMBER']])

    res = flatten(units[::-1])

    if len(res) < 5:
        res += [''] * (5 - len(res))
    elif len(res) > 5:
        res = res[:5]

    return pd.Series(res)


def split_postcode_uprn(row):

    p = re.compile(r'([A-Z]{1,2})([0-9][A-Z0-9]?)(?: +)?([0-9])([A-Z])([A-Z])')

    match = re.match(p, row['POSTCODE'])

    pc0, pc1 = ''.join([match.group(i) for i in range(1, 3)]), ''.join([match.group(i) for i in range(3, 6)])

    pc_area, pc_district, pc_sector, pc_unit_0, pc_unit_1 = [match.group(i) for i in range(1, 6)]

    return pd.Series([pc0, pc1, pc_area, pc_district, pc_sector, pc_unit_0, pc_unit_1])


def count_n_tenement(df):

    df['n_tenement'] = 1

    groups = []

    for (i, j), group in df.groupby(['BUILDING_NAME', 'POSTCODE']):
        if (i != '') and (len(group) > 1):
            group['n_tenement'] = len(group)

        groups.append(group)

    df = pd.concat(groups)

    groups = []

    for (i, j), group in df.groupby(['BUILDING_NUMBER', 'POSTCODE']):
        if (i != '') and (len(group) > 1):
            group['n_tenement'] = len(group)
        groups.append(group)

    df = pd.concat(groups)

    return df

def indexing_uprn(df):

    df = count_n_tenement(df)

    df['type_of_micro'] = df.apply(type_of_micro, axis=1)
    df['type_of_macro'] = df.apply(type_of_macro, axis=1)

    df[['number_like_%s' % i for i in range(5)]] = df.apply(parse_number_like_uprn, axis=1)
    df[['pc0', 'pc1', 'pc_area', 'pc_district', 'pc_sector', 'pc_unit_0', 'pc_unit_1']] = \
        df.apply(split_postcode_uprn, axis=1)

    return df


# dm = SqlDBManager()
#
# dm.list_global_db()
#
# sql_db = dm.get_db(db_name='the_database')
#
# sql_db.setup_database(if_exist='replace')
#
# sql_db.sql_query("select * from indexed")
