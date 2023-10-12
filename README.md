# FLAP

*FLAP* is an open-source tool for linking free-text addresses to 
Ordinance Survey Unique Property Reference Number (OS UPRN). You need to have a
licence of OS UPRN and download the address premium product to use *FLAP*
*FLAP* can be used at scale with a few lines of syntax

# Quick start of FLAP tool

*Note: The Tool is tested on Ubuntu 22*

### Step 1: Clone this repo
Please refer to instructions of `git`

### Step 2: Setup the environment
Using the `requirement.txt` file

Please refer to instructions of python `venv` and `pip`

### Step 3: Create and compile the database

*Note: All python script should be run from root of the repo folder as the 
current working directory (referred to as `.` in the following).*

#### Create a database

The database creation is handled by the `src.database.sql.SqlDBManager` class

```python
import os
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([os.getcwd()])

from src.database.sql import SqlDBManager

sql_dbm = SqlDBManager()

sql_dbm.create_global_db('the_database')
```

The script will create directories 
`./db/the_database` and `./db/the_database/raw`

Next, paste the `.zip` file from the OS UPRN product to `./db/the_database/raw`, 
so that we are ready to compile the database

#### Compile the database

The database operations are handled with `src.database.sql.SqlDB` class

```python
import os
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([os.getcwd()])

from src.database.sql import SqlDB

sql_db = SqlDB('./db/the_database')

sql_db.setup_database()
```

It takes several minutes to finish depending on the volume 
of the database and your hardware specs


#### To be continued