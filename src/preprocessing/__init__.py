import re
import warnings

from src.utils import repl_positions


def is_postcode(string):
    postcode_regex = r'^([A-Z]{1,2})([0-9][A-Z0-9]?) ?([0-9][A-Z]{2})$'
    if isinstance(string, str):
        return bool(re.match(postcode_regex, string))
    else:
        return False


def postcode_preproc(string, suppress_warnings=False):
    postcode_regex = r'^([A-Z]{1,2})([0-9][A-Z0-9]?) ?([0-9][A-Z]{2})$'

    if is_postcode(string):
        return re.sub(pattern=postcode_regex, repl=r'\1\2 \3', string=string)
    else:
        if not suppress_warnings:
            warnings.warn('%s is not a postcode' % string)
        return string


def address_line_preproc(address):
    preproc_mapping = [
        (r'[^\w\-/\s]', ' '),
        (r'  +', ' '),
        (r'--+', '-'),
        (r'//+', '/')
    ]

    n_sub = 0
    n_abbr = 0
    add_pt = 0
    add_dl = 0
    remove_region = 0

    for pattern, repl in preproc_mapping:
        address, n = re.subn(pattern=pattern, repl=repl, string=address)
        n_sub += n

    address = address.upper()

    log = {'address_preproc': address, 'preproc_n_sub': n_sub, 'preproc_n_abbr': n_abbr, 'add_pt': add_pt,
           'add_dl': add_dl, 'remove_region': remove_region}

    return address, log
