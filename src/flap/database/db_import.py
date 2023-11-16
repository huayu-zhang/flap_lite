"""
DB_read module extract delivery point address from Address Premium Product

Currently, it accepts two types: zip of zips of csvs (.zip) or a geopackage (.gpkg)

A 'raw' table will be created in the Sqlite database
"""

from zipfile import ZipFile
import csv
import io
import pandas as pd
import geopandas as gpd
import os
from tqdm import tqdm


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
            df.loc[:, col] = df[col].apply(lambda x: numeric_to_object(x))

        else:
            df.loc[:, col] = df[col]

    return df


def db_import(sql_db):

    zip_path = [os.path.join(sql_db.sub_paths['raw'], file)
                for file in os.listdir(sql_db.sub_paths['raw']) if '.zip' in file][0]

    sql_db.drop_table('raw')

    conn = sql_db.get_conn()

    with ZipFile(zip_path, 'r') as z:

        zz_files = [zz_file for zz_file in z.namelist()]

        if any('.gpkg' in zz_file for zz_file in zz_files):

            print('.gpkg database found. Starting to read db file')

            zz_data = io.BytesIO(z.read(zz_files[0]))

            gpkg_db = gpd.read_file(zz_data, layer='delivery_point_address')

            print('DB reading complete, start saving to SQL')

            gpkg_db = convert_df_to_object(gpkg_db)

            cols_upper = [col.upper() for col in gpkg_db.columns]

            gpkg_db.columns = cols_upper

            gpkg_db.to_sql(name='raw', con=conn, if_exists='append', dtype='TEXT', index=False)

            print('Raw table building finished')

        elif any('.zip' in zz_file for zz_file in zz_files):
            print('.zip database found. Starting to read db file')

            chunk_size = int(1e5)
            rows = []
            data_index = '28'

            sql_config = sql_db.get_db_config()

            zz_files = [zz_file for zz_file in z.namelist() if '.zip' in zz_file]

            for zz_file in tqdm(zz_files):
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
