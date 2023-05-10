# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:37:43 2022

@author: hzhang3410

Expansion of UPRN databases

- BUILDING_NAME Range -> BUILDING_NUMBER or BUILDING_NAME expansion

- SUB_BUILDING_NAME + BUILDING_NUMBER -> BUILDING_NAME Alternative expansion
- BUILDING_NAME -> SUB_BUILDING_NAME + BUILDING_NUMBER Alternative expansion

- THOROUGHFARE, DEPENDENT_THOROUGHFARE abbreviation expansion
- BUILDING_NAME, SUB_BUILDING_NAME abbreviation expansion
- POST_TOWN abbreviation expansion

Make sure that there are no conflicts with original records

"""
import re
import itertools
import pandas as pd


uprn_street_synonyms = {
                "ROAD": ["RD"],
                "STREET": ["ST"],
                "PLACE": ["PL"],
                "CRESCENT": ["CRES"],
                "DRIVE": ["DR"],
                "AVENUE": ["AVE"],
                "TERRACE": ["TER"],
                "GARDENS": ["GDNS"],
                "GARDEN": ["GDN"],
                "COURT": ["CT"],
                "PARK": ["PK"],
                "ST.": ["ST"],
                "GROVE": ["GR", "GRO"],
                "SOUTH": ["S"],
                "WEST": ["W"],
                "NORTH": ["N"],
                "WALK": ["WLK"],
                "EAST": ["E"],
                "SQUARE": ["SQ"],
                "CLOSE": ["CL"],
                "CHURCH": ["CH"],
                "SCHOOL": ["SCH"],
                # "FLAT": ["FLT", "FL"],
                "APPARTMENT": ["APT"],
                "APPARTMENTS": ["APTS"],
                "FARM": ["FM"],
                "HOUSE": ["HSE"],
                "ACCOMMODATION": ["ACC"],
                "LIEUTENANT": ["LT"],
                "LODGE": ["LDG"],
                "COTTAGE": ["COTT"]
}


def parse_range(text):
    p = re.compile(r'^([A-Z ]+)?(\d+)([A-Z])?-(\d+)([A-Z])?([A-Z ]+)?$')

    match = re.match(p, text)

    if match is None:
        return None

    else:
        parsing_seq = ['prefix', 'num_0', 'char_0', 'num_1', 'char_1', 'suffix']
        parsed = {name: group for name, group in zip(parsing_seq, match.groups())}
        return parsed


def range_letter_upper(letter_1, letter_2):
    l2n = {chr(x): x for x in range(65, 91)}
    return [chr(x) for x in range(l2n[letter_1], l2n[letter_2] + 1)]


def expand_str(ll):
    ll = [l for l in ll if len(l)]
    return [''.join(i) for i in itertools.product(*ll)]


def bn_range_to_list(row):

    text = row['BUILDING_NAME']

    parsed = parse_range(text)

    if parsed is None:
        return []

    else:
        pre_expand = {
            'prefix': [] if parsed['prefix'] is None else [parsed['prefix']],
            'range': [],
            'suffix': [] if parsed['suffix'] is None else [parsed['suffix']]
        }

        interval = 1 if ((int(parsed['num_0']) + int(parsed['num_1'])) % 2) or \
                        (parsed['prefix'] is not None) or \
                        (parsed['suffix'] is not None) else 2

        if (parsed['char_0'] is None) and (parsed['char_1'] is None):

            pre_expand['range'] = [str(i)
                                   for i in range(int(parsed['num_0']), int(parsed['num_1']) + interval, interval)]

        elif parsed['char_0'] is None:

            if parsed['num_0'] == parsed['num_1']:
                pre_expand['range'] = [parsed['num_0'], ''.join([parsed['num_1'], parsed['char_1']])]
            else:
                pre_expand['range'] = [
                    str(i)
                    for i in range(int(parsed['num_0']), int(parsed['num_1']), interval)
                    ] + [''.join([parsed['num_1'], parsed['char_1']])]

        elif (parsed['char_0'] is not None) and (parsed['char_1'] is not None):

            if parsed['num_0'] == parsed['num_1']:

                pre_expand['range'] = [
                    ''.join([parsed['num_0'], letter])
                    for letter in range_letter_upper(parsed['char_0'], parsed['char_1'])
                ]

        expand_mappings = []

        if len([pre_expand['range']]):

            for s in expand_str(list(pre_expand.values())):
                if re.search(r'[A-Z]', s) is not None:
                    expand_mappings.append({'BUILDING_NAME': s})
                else:
                    expand_mappings.append({'BUILDING_NAME': '', 'BUILDING_NUMBER': s})

        if isinstance(row, dict):
            ref_row_d = row
        else:
            ref_row_d = row.to_dict()

        rows = []

        for mapping in expand_mappings:
            rows.append(dict(ref_row_d,
                             ex_label='-'.join([ref_row_d['ex_label'], '1rng']),
                             **mapping))

    return rows


def parse_bn(text):

    p = re.compile(r'^(\d+)([A-Z]|/\d+)$')

    match = re.match(p, text)

    if match is None:
        return None

    else:
        return {'num_0': match.group(1), 'num_char_1': match.group(2).replace('/', '')}


def bn_to_bnu_sbn_list(row):

    if len(row['SUB_BUILDING_NAME']) or len(row['BUILDING_NUMBER']):
        return []

    text = row['BUILDING_NAME']

    parsed = parse_bn(text)

    if parsed is None:
        return []

    else:
        expand_mappings = [{'SUB_BUILDING_NAME': 'FLAT %s' % parsed['num_char_1'],
                            'BUILDING_NAME': '',
                            'BUILDING_NUMBER': parsed['num_0']}]

        if isinstance(row, dict):
            ref_row_d = row
        else:
            ref_row_d = row.to_dict()

        rows = []

        for mapping in expand_mappings:
            rows.append(dict(ref_row_d,
                             ex_label='-'.join([ref_row_d['ex_label'], '2al1']),
                             **mapping))

    return rows


def bnu_sbn_to_bn_list(row):

    bnu = str(row['BUILDING_NUMBER'])
    bn = row['BUILDING_NAME']
    sbn = row['SUB_BUILDING_NAME']

    if len(bnu) and len(sbn) and not len(bn):

        p = re.compile(r'(FLAT|UNIT) (\d+[A-Z]?|[A-Z])')
        match = re.search(p, sbn)

        if match is None:
            return []

        else:
            if re.search(r'\d', match.group(2)) is None:
                bn_new = ''.join([bnu, match.group(2)])
            else:
                bn_new = '/'.join([bnu, match.group(2)])

            if len(bn):
                bn_new += ' ' + bn

            expand_mappings = [{'SUB_BUILDING_NAME': '', 'BUILDING_NAME': bn_new, 'BUILDING_NUMBER': ''}]

            if isinstance(row, dict):
                ref_row_d = row
            else:
                ref_row_d = row.to_dict()

            rows = []

            for mapping in expand_mappings:
                rows.append(dict(ref_row_d,
                                 ex_label='-'.join([ref_row_d['ex_label'], '2al2']),
                                 **mapping))

        return rows

    else:
        return []


def sbn_bn_bnu_alt(row):
    return bn_to_bnu_sbn_list(row) + bnu_sbn_to_bn_list(row)


def tokenize_by_re(pattern, s):
        if isinstance(s, str):
            return [token for token in re.split(pattern, s) if len(token)]
        else:
            return ['non_str']


def tokenize_default(s):
    pattern = r'( )'
    return tokenize_by_re(pattern, s)


def expand_list(ll):
    return [list(item) for item in itertools.product(*ll)]


def expand_dict(d):
    ll = expand_list(d.values())
    dl = []
    
    for l in ll:
        dl.append({k:v for k, v in zip(d.keys(), l)})
    
    return dl


def abbr_to_list(row):

    pre_expand = {
                    'SUB_BUILDING_NAME': [],
                    'BUILDING_NAME': [],
                    'DEPENDENT_THOROUGHFARE': [],
                    'THOROUGHFARE': [],
                    'POST_TOWN': [],
                    'DEPENDENT_LOCALITY': [],
                    'DOUBLE_DEPENDENT_LOCALITY': []
                  }
    
    for key in pre_expand:
        tokens = tokenize_default(row[key])
        token_syns = [] 
        
        for token in tokens:
            if token in uprn_street_synonyms.keys():
                token_syns.append([token] + uprn_street_synonyms[token])
            else:
                token_syns.append([token])
        pre_expand[key] = expand_str(token_syns)
    
    expand_mappings = expand_dict(pre_expand)[1:]

    if isinstance(row, dict):
        ref_row_d = row
    else:
        ref_row_d = row.to_dict()

    rows = []

    for mapping in expand_mappings:
        rows.append(dict(ref_row_d,
                         ex_label='-'.join([ref_row_d['ex_label'], '3syn']),
                         **mapping))

    return rows


def expand_row(iterrow):

    d_func = {
        'rng': bn_range_to_list,
        'alt': sbn_bn_bnu_alt,
        'syn': abbr_to_list
    }

    i, row = iterrow

    ref_row = row.to_dict()
    ref_row['ex_label'] = '0'

    indices = [i]
    rows = [ref_row]

    for key, func in d_func.items():

        expand_rows = list(itertools.chain(*[func(row) for row in rows]))

        if len(expand_rows):
            rows.extend(expand_rows)
            for j, _ in enumerate(expand_rows):
                indices.append('-'.join([str(i), key, str(j)]))

    return indices, rows


def make_ex_uprn(row):
    return '_'.join([row['UPRN'], row['ex_label']])


COLUMNS = ["RECORD_IDENTIFIER", "CHANGE_TYPE", "PRO_ORDER", "UPRN", "UDPRN", "ORGANISATION_NAME", "DEPARTMENT_NAME",
           "SUB_BUILDING_NAME", "BUILDING_NAME", "BUILDING_NUMBER", "DEPENDENT_THOROUGHFARE", "THOROUGHFARE",
           "DOUBLE_DEPENDENT_LOCALITY", "DEPENDENT_LOCALITY", "POST_TOWN", "POSTCODE", "POSTCODE_TYPE",
           "DELIVERY_POINT_SUFFIX", "WELSH_DEPENDENT_THOROUGHFARE", "WELSH_THOROUGHFARE",
           "WELSH_DOUBLE_DEPENDENT_LOCALITY", "WELSH_DEPENDENT_LOCALITY", "WELSH_POST_TOWN", "PO_BOX_NUMBER",
           "PROCESS_DATE", "START_DATE", "END_DATE", "LAST_UPDATE_DATE", "ENTRY_DATE", "ex_label", "ex_uprn"]


def expand_uprn(df, duplicate_check_columns=None):

    if duplicate_check_columns is None:
        duplicate_check_columns = COLUMNS[5:16]
        
    l = map(expand_row, list(df.iterrows()))

    index_list, record_list = zip(*l)

    records = list(itertools.chain(*record_list))
    indices = list(itertools.chain(*index_list))

    df_ex = pd.DataFrame.from_records(records, index=indices)

    df_ex.sort_values(by='ex_label', inplace=True)

    if duplicate_check_columns is not None:
        df_ex.drop_duplicates(subset=duplicate_check_columns, inplace=True)

    df_ex['ex_uprn'] = df_ex.apply(make_ex_uprn, axis=1)

    return df_ex
