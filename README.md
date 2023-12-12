![PyPI](https://img.shields.io/pypi/v/flap-lite?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/flap-lite)

# FLAP

*FLAP* is an open-source tool for linking free-text addresses to 
Ordinance Survey Unique Property Reference Number (OS UPRN). You need to have a
licence of OS UPRN and download the address premium product to use *FLAP*
*FLAP* can be used at scale with a few lines of syntax.


# Quick start of FLAP tool

## Installation

We recommend you to create a virtual environment with `venv`. 

```shell
python3 -m venv [YOUR_PATH]/flap_lite
source [YOUR_PATH]/flap_lite/bin/activate
```

Install with `pip`: 
```shell
pip install --upgrade flap-lite
```

For now, please contact the developer for downloading the trained model. Copy the model to
`[YOUR_PATH]/flap_lite/lib/python3.9/site-packages/flap/model/`

```shell
cp [PATH_TO_MODEL_FILE] [YOUR_PATH]/flap_lite/lib/python3.9/site-packages/flap/model/
```

## Quick Start

### Building the database

Use `flap.create_database` for building the database.

```python
from flap import create_database

create_database(db_path=[PATH_FOR_THE_DB], raw_db_file=[PATH_TO_DB_ZIP])
```

### Matching

Use `flap.match` for matching address to database

```python
from flap import match

input_csv = '[PATH_TO_INPUT_CSV_FILE]'
db_path = '[PATH_TO_THE_DB]'

results = match(
    input_csv=input_csv,
    db_path=db_path
    )
```

Matching results will be saved to `[$pwd]/output.csv` by default. By default, *FLAP* uses all available CPUs and 
process the addresses in batches of 10,000. 

Some useful options are: 
* `batch_size` for number of addresses in each batch
* `max_workers` for CPU cores used
* `in_memory_db` for if in-memory SQLite is used

# How does it work?


Briefly, *FLAP* parses the structured parts of addresses (e.g. POSTCODE "AB12 3CD"). 
And all the deterministic parts (e.g. numbers "111", letters "A")

An SQL query is made based on the parsed fields to narrow down to a few rows in the database. 
```sqlite
select * from indexed where POSTCODE='AB12 3CD'
```

Features are generated:
* For the deterministic parts: pairwise comparison to see if equal
* Linear assignment alignment for the textual parts
* For postcode: comparison to see if parts are equal

A trained *Random Forest Classifier* predict a score based on the generated feature. The address with best score is 
deemed as a match. 

The above is a simplified description.

# Coming soon

- [ ] Command Line Interface
- [ ] More documentation
- [ ] Dummy database for trying it out with an example notebook
