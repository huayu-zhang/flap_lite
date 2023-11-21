from flap.database.sql import SqlDB, SqlDBManager
from shutil import copy


def create_database(db_path, raw_db_file):

    dm = SqlDBManager()
    dm.create_db(db_path)

    sql_db = SqlDB(db_path)

    copy(raw_db_file, sql_db.sub_paths['raw'])

    sql_db.build()
