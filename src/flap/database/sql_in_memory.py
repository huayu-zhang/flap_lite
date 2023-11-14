import sqlite3
import pandas as pd


class SqlDBInMemory:

    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.columns_dict = {}

    def load_csv(self, path, table_name, chunksize=10000):

        print('Start loading DB to memory')
        db_chunks = pd.read_csv(path, chunksize=chunksize, dtype='object')

        i = 0

        while 1:
            try:
                df = next(db_chunks)
                i += 1
                df.fillna('', inplace=True)
                df.to_sql(name=table_name, con=self.conn, if_exists='append', dtype='TEXT', index=False)
                print('Chunk %s loaded' % i, end='\r')

            except StopIteration:
                break

        self.columns_dict[table_name] = df.columns.to_list()

        print('DB Loading Finished')

    def get_columns_of_table(self, table_name):
        return self.columns_dict[table_name]

    def sql_query(self, query):

        cur = self.conn.cursor()
        cur.execute(query)
        res = cur.fetchall()
        cur.close()

        return res

    def create_index(self, table_name, columns):

        index_name = '__'.join(columns)

        sql = "CREATE INDEX IF NOT EXISTS %s ON %s(%s)" % (
            index_name,
            table_name,
            ', '.join(columns)
        )

        cur = self.conn.cursor()
        cur.execute(sql)
        cur.close()

    def close(self):
        self.conn.close()
