import os
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([os.getcwd()])

import argparse
from src.database.sql import SqlDBManager, SqlDB


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('database_name', help='Name of the database')
    parser.add_argument('--position', default='global',
                        help="""Pass a "[PROJECT NAME]" for project level database, 
                        or "global" for global database. Default "global" """)
    parser.add_argument('--action', default='create',
                        help="""{create, build} Use create first to create the db_path,
                        and then copy the raw file under "[db_path]/raw". Run build then.
                        Default 'create'. 
                        """)
    parser.add_argument('--if_exist', default='skip',
                        help="""{skip, replace} Whether to skip if the database existed or to replace
                        """)
    return parser


if __name__ == '__main__':
    args = arg_parser().parse_args()

    if args.position == 'global':
        dbm = SqlDBManager()
        dbm.create_global_db(args.database_name)
        path_to_db = os.path.join(dbm.global_db_path, args.database_name)
        sql_db = SqlDB(path_to_db)

    else:
        dbm = SqlDBManager(project_name=args.position)
        dbm.create_project_db(args.database_name)
        path_to_db = os.path.join(dbm.project_db_path, args.database_name)
        sql_db = SqlDB(path_to_db)

    if args.action == 'build':
        sql_db.setup_database()
