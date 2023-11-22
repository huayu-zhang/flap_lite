"""
This is the top level API for building the database required for FLAP.
The raw database file can be either a zip file of zips of csv files or a zip of gpkg file.
The delivery point address will be extracted and a SQLite database created to be used by matching.
"""


from flap.database.sql import SqlDB, SqlDBManager
from shutil import copy


def create_database(db_path, raw_db_file):
    """
    Creates the SQLite database and index the database from zip files download from OS ABP. The raw database file can be
    either a zip file of zips of csv files or a zip of gpkg file. The delivery point address will be extracted and a
    SQLite database created to be used by matching.

    Parameters
    ----------
    db_path: str
        the directory for the database
    raw_db_file
        the path to the raw zip file
    Returns
    -------
        None

    Examples
    --------
    >>> from flap import create_database
    >>> create_database([db_path], [raw_db_file])
    """

    dm = SqlDBManager()
    dm.create_db(db_path)

    sql_db = SqlDB(db_path)

    copy(raw_db_file, sql_db.sub_paths['raw'])

    sql_db.build()
