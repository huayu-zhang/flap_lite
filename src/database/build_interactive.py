import os
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([os.getcwd()])

import argparse
from src.database.sql import SqlDBManager, SqlDB


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', help='The name of project')
    return parser


if __name__ == '__main__':

    print('Now lets setup the database')
    args = arg_parser().parse_args()

    database_name = input('Please enter the name of the database: ')
    project_specific = input('Is the database project specific [yes/no]: ')

    if project_specific == 'no':
        dbm = SqlDBManager()
        dbm.create_global_db(database_name)
        path_to_db = os.path.join(dbm.global_db_path, database_name)
        sql_db = SqlDB(path_to_db)

    else:
        dbm = SqlDBManager(project_name=args.project_name)
        dbm.create_project_db(database_name)
        path_to_db = os.path.join(dbm.project_db_path, database_name)
        sql_db = SqlDB(path_to_db)

    print('A database folder have been at %s' % path_to_db)
    input('Please copy the database zip file to %s, and then press <<ENTER>>' % os.path.join(path_to_db, 'raw'))

    if sql_db.db_status['table_raw_built']:
        replace = input('The database exists, would you like to replace the existing one [yes/no]: ')
        if replace == 'yes':
            sql_db.build_raw(if_exist='replace')
        else:
            print('Database build skipped.')
    else:
        sql_db.build_raw()
