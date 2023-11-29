import sqlite3
import pandas as pd


class SqlDBInMemory:

    def __init__(self):
        self.columns_dict = {}

    @staticmethod
    def get_conn():
        return sqlite3.connect('file:cachedb?mode=memory&cache=shared', timeout=10)

    def load_csv(self, path, table_name, chunksize=10000):

        print('Start loading DB to memory')
        db_chunks = pd.read_csv(path, chunksize=chunksize, dtype='object', index_col=0)

        i = 0

        while 1:
            try:
                df = next(db_chunks)
                i += 1
                df.fillna('', inplace=True)
                df.to_sql(name=table_name, con=self.get_conn(), if_exists='append', dtype='TEXT', index=False)
                print('Chunk %s loaded' % i, end='\r')

                self.columns_dict[table_name] = df.columns.to_list()

            except StopIteration:
                break

        print('DB Loading Finished')

    def get_columns_of_table(self, table_name):
        return self.columns_dict[table_name]

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

    def create_index(self, table_name, columns):

        index_name = '__'.join(columns)

        sql = "CREATE INDEX IF NOT EXISTS %s ON %s(%s)" % (
            index_name,
            table_name,
            ', '.join(columns)
        )

        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    def close(self):
        conn = self.get_conn()
        conn.close()
