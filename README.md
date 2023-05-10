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

### Step 4: Create a project and copy in input data

#### Create a project

Projects are managed using `src.project.project.ProjectManager` class

```python
import os
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([os.getcwd()])

from src.project.project import ProjectManager

project_manager = ProjectManager()
project_manager.create_project('the_project')
```

The script will create the directory `./projects/the_project`
and several sub-directories including `./projects/the_project/input`

#### Copy in the input data

The input data should be a `.csv` file with a index column 
and two data columns: `input_id` and `input_address`

|   | input_id | input_address                     |
|---|----------|-----------------------------------|
| 0 | abc123   | 1 Fairy Rd, Fairyland, FL99 9XX   |
| 1 | def456   | 2 Angel Lane, Fairyland, FL88 8XX |

Or in plain `.csv`

```csv
,input_id,input_address
0,abc123,"1 Fairy Rd, Fairyland, FL99 9XX"
1,def456,"2 Angel Lane, Fairyland, FL88 8XX"
```

Copy the input `.csv` file to `./projects/the_project/input` and 
name it `input_df.csv`

### Step 5: Match the data to UPRN DB

#### Matching
Matching of the project is managed by `src.project.project.Project` class

```python
from src.database.sql import SqlDB
from src.project.project import Project


project = Project('./projects/the_project')

sql_db = SqlDB('./db/the_database')

project.match(sql_db=sql_db)
```

#### Getting the results
After running the above script, you can find the matching 
results in `./projects/the_project/output_formatted/results.csv`

The output is a `.csv` file in the format of 

|   | input_id | UPRN    |
|---|----------|---------|
| 0 | abc123   | 1000111 |
| 1 | def456   | 1000112 |

If you would like detailed output, you can find the summary in
`./projects/the_project/output_summary/output_summary*.csv`

The output summary is in the format of

|   | task_id | input_address | uprn_fields | input_id | db_id | UPRN | error |
|---|---------|---------------|-------------|----------|-------|------|-------|
| 0 |         |               |             |          |       |      |       |
| 1 |         |               |             |          |       |      |       |


### [OPTIONAL] If you like to use the low-level functionality

A minimal example is:

```python
from src.pipelines.tree_align import RuleMatchByTree
from src.database.sql import SqlDB

sql_db = SqlDB('./db/the_database')

matcher = RuleMatchByTree(sql_db)

matcher.match(address='1 Fairy Rd, Fairyland, FL99 9XX')

print(matcher.get_database_id())
print(matcher.log)
```

# Other information
**Full documentations and functionality will be added in due time**