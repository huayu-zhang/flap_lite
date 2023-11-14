import re
import warnings

from flap.utils import repl_positions


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
    if not isinstance(address, str):
        return address, {'address_preproc': address, 'proceed': False}
    elif len(address) == 0:
        return address, {'address_preproc': address, 'proceed': False}

    preproc_mapping = [
        (r"[^\w\-/\s'\.\(\)]", ' '),
        (r'  +', ' '),
        (r'--+', '-'),
        (r'//+', '/')
    ]

    n_sub = 0
    # n_abbr = 0
    # add_pt = 0
    # add_dl = 0
    # remove_region = 0

    for pattern, repl in preproc_mapping:
        address, n = re.subn(pattern=pattern, repl=repl, string=address)
        n_sub += n

    address = address.upper()

    #
    # parsed = rule_parsing(address)
    # address_rp = parsing_to_string(parsed, address)
    #
    #
    # if len(parsed['REGION']):
    #     start, end = parsed['REGION'][0]
    #     address = repl_positions(address, list(range(start, end)))
    #     remove_region += 1
    #
    # if not len(parsed['POST_TOWN']):
    #
    #     parsed = rule_parsing(address)
    #
    #     if len(parsed['DEPENDENT_LOCALITY']) and len(parsed['POSTCODE']):
    #         start_dl, end_dl = parsed['DEPENDENT_LOCALITY'][0]
    #         start_pc, end_pc = parsed['POSTCODE'][0]
    #         dlpc0 = ':'.join([address[start_dl:end_dl],
    #                           address[start_pc:end_pc].split(' ')[0]])
    #
    #         if dlpc0 in dlpc0_to_pt:
    #             address = ' '.join([address[:end_dl], dlpc0_to_pt[dlpc0],
    #                                 address[end_dl:]])
    #
    #             add_pt += 1
    #
    #     elif len(parsed['DOUBLE_DEPENDENT_LOCALITY']) and len(parsed['POSTCODE']):
    #         start_ddl, end_ddl = parsed['DOUBLE_DEPENDENT_LOCALITY'][0]
    #
    #         if address[start_ddl:end_ddl] in ddl_to_dlpc0:
    #             dlpc0 = ddl_to_dlpc0[address[start_ddl:end_ddl]]
    #             pt = dlpc0_to_pt[dlpc0]
    #
    #             address = ' '.join([address[:end_ddl],
    #                                 dlpc0.split(':')[0], pt, address[end_ddl:]])
    #
    #             add_pt += 1
    #             add_dl += 1
    #
    # address, _ = re.subn(r' +', ' ', address.replace('*', '').upper())
    #
    log = {'address_preproc': address, 'preproc_n_sub': n_sub, 'proceed': True
           # 'preproc_n_abbr': n_abbr, 'add_pt': add_pt,
           # 'add_dl': add_dl, 'remove_region': remove_region
           }

    return address, log
