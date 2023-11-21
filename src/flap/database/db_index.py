"""
Process of DB index:

Query raw db in chunks

Create Range expansion

Index raw data and range expansion

Expansion

"""

import pandas as pd
from tqdm import tqdm
import re
import itertools
import numpy as np

from flap.utils import flatten


def db_index(sql_db):

    print('Start indexing and augmentation')

    conn_temp = sql_db.get_conn_temp()

    tasks_gen = sql_db.sql_table_batch_by_column('raw', 'POSTCODE', int(1e4))

    for df_chunk in tqdm(tasks_gen):

        # Process chunks of raw data and save to 'indexed' table
        df_chunk = df_chunk.drop_duplicates()
        df_chunk['ex_label'] = '0'

        res_indexed = indexing_uprn(df_chunk)
        res_indexed.to_sql(name='indexed', con=conn_temp, if_exists='append', dtype='TEXT', index=False)

        # get and index range
        df_chunk_range = expand_bn(df_chunk)
        res_range_indexed = indexing_uprn(df_chunk_range)

        # add range
        res_both = pd.concat([res_indexed, res_range_indexed])
        res_both.reset_index(inplace=True)

        # expand both and save to 'expanded'
        res_expanded = expand_number_like(res_both)
        res_expanded.to_sql(name='expanded', con=conn_temp, if_exists='append', dtype='TEXT', index=False)

    # Saving to main database

    for table_name in ['indexed', 'expanded']:

        print('Saving to %s' % table_name)

        sql_chunk_gen = pd.read_sql(sql='SELECT * FROM %s' % table_name, con=conn_temp, chunksize=int(1e5))

        conn = sql_db.get_conn()

        for chunk in tqdm(sql_chunk_gen):
            chunk.to_sql(name=table_name,
                         con=conn,
                         if_exists='append',
                         dtype='TEXT',
                         index=False
                         )

    conn.close()
    conn_temp.close()

    print('Finished database indexing')


# Below are indexing functions

NUMBER_LIKE_MASTER_REGEX = re.compile(
    r"(FLAT|UNIT|BUILDING|ROOM|BLOCK|BONDS?|FL|APARTMENT|\(F|F|-|\()? ?"
    r"((\dF\d+)|(\d+[A-Z]\d+)|([A-EG-Z])?(\d+)([A-Z])?|(?<!')(^|\b)([A-Z]|PF\d?|BF\d?|GF\d?)($|\b)(?!'))"
    r"\)?")


sbn_det_dict = {
    'level_det': ['BASEMENT', 'BOTTOM', 'GROUND', 'LOWER', 'FIRST', 'SECOND', 'THIRD',
                  'FOURTH', 'MIDDLE', 'MID', 'UPPER', 'ATTIC', 'TOP'],
    'seperator': ['APARTMENT', 'FLOOR', 'FLAT'],
    'flat_det': ['FRONT', 'REAR', 'LEFT', 'RIGHT', 'CENTRE']
}

sbn_det_tokens = []

for k, v in sbn_det_dict.items():
    sbn_det_tokens.extend(v)


sbn_parse_rule = {
    'level_det': r"%s" % '|'.join(sbn_det_dict['level_det']),
    'flat_det': r"%s" % '|'.join(sbn_det_dict['flat_det'])
}


def type_of_macro(row):

    cols = ['DEPENDENT_THOROUGHFARE', 'THOROUGHFARE', 'DOUBLE_DEPENDENT_LOCALITY', 'DEPENDENT_LOCALITY']

    return ''.join([str(int(len(row[col]) > 0)) for col in cols])


def type_of_micro(row):

    cols = ['ORGANISATION_NAME', 'DEPARTMENT_NAME', 'SUB_BUILDING_NAME',
            'BUILDING_NAME', 'BUILDING_NUMBER']

    return ''.join([str(int(len(row[col]) > 0)) for col in cols])


def parse_sbn_det(s):

    m_level = re.search(sbn_parse_rule['level_det'], s)
    m_flat = re.search(sbn_parse_rule['flat_det'], s)

    try:
        level = m_level.group(0)
    except AttributeError:
        level = ''

    try:
        flat = m_flat.group(0) if m_flat is not None else ''

    except AttributeError:
        flat = ''

    return level, flat


def all_tokens_in_all(s):
    list_of_tokens = re.split(r' ', s)
    return all([t in sbn_det_tokens for t in list_of_tokens])


def parse_number_like_uprn(row):

    p = NUMBER_LIKE_MASTER_REGEX

    split_pattern = re.compile(r"(?<=\d)(?=\D)|(?=\d)(?<=\D)")

    units = list()

    if len(row['SUB_BUILDING_NAME']):
        matches = list(re.finditer(p, row['SUB_BUILDING_NAME']))
        units.append(
            flatten([re.split(split_pattern, match.group(2))
                     if re.search(r'\dF\d', match.group(2)) is None else [match.group(2)]
                     for match in matches]))

        if all_tokens_in_all(row['SUB_BUILDING_NAME']):
            level, flat = parse_sbn_det(row['SUB_BUILDING_NAME'])
            units.append([level, flat])

    if len(row['BUILDING_NAME']):
        matches = list(re.finditer(p, row['BUILDING_NAME']))
        units.append(
            flatten([re.split(split_pattern, match.group(2))
                     if re.search(r'\dF\d', match.group(2)) is None else [match.group(2)]
                     for match in matches]))

        if all_tokens_in_all(row['SUB_BUILDING_NAME']):
            level, flat = parse_sbn_det(row['SUB_BUILDING_NAME'])
            units.append([level, flat])

    if len(row['BUILDING_NUMBER']):
        units.append([row['BUILDING_NUMBER']])

    res = flatten(units[::-1])

    if len(res) < 5:
        res += [''] * (5 - len(res))
    elif len(res) > 5:
        res = res[:5]

    return pd.Series(res)


def split_postcode_uprn(row):

    p = re.compile(r'([A-Z]{1,2})([0-9][A-Z0-9]?)(?: +)?([0-9])([A-Z])([A-Z])')

    match = re.match(p, row['POSTCODE'])

    pc0, pc1 = ''.join([match.group(i) for i in range(1, 3)]), ''.join([match.group(i) for i in range(3, 6)])

    pc_area, pc_district, pc_sector, pc_unit_0, pc_unit_1 = [match.group(i) for i in range(1, 6)]

    return pd.Series([pc0, pc1, pc_area, pc_district, pc_sector, pc_unit_0, pc_unit_1])


def count_n_tenement(df):

    df['n_tenement'] = 1

    groups = []

    for (i, j), group in df.groupby(['BUILDING_NAME', 'POSTCODE']):
        if (i != '') and (len(group) > 1):
            group['n_tenement'] = len(group)

        groups.append(group)

    df = pd.concat(groups)

    groups = []

    for (i, j), group in df.groupby(['BUILDING_NUMBER', 'POSTCODE']):
        if (i != '') and (len(group) > 1):
            group['n_tenement'] = len(group)
        groups.append(group)

    df = pd.concat(groups)

    return df


def indexing_uprn(group):

    group = count_n_tenement(group)

    group['type_of_micro'] = group.apply(type_of_micro, axis=1)
    group['type_of_macro'] = group.apply(type_of_macro, axis=1)

    group[['number_like_%s' % i for i in range(5)]] = group.apply(parse_number_like_uprn, axis=1)
    group[['pc0', 'pc1', 'pc_area', 'pc_district', 'pc_sector', 'pc_unit_0', 'pc_unit_1']] = \
        group.apply(split_postcode_uprn, axis=1)

    return group


# Code below is for range expansion


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


def expand_bn(group):

    expand_collections = []

    for _, row in group.iterrows():
        ex = bn_range_to_list(row)
        expand_collections.extend(ex)

    return pd.DataFrame.from_records(expand_collections)


# Code below is for expansion

# # Ranking and Mapping functions


def parse_multiplier(x):
    m = re.search(r'(\d+)F(\d+)', x)
    level = int(m.group(1))
    flat = int(m.group(2))
    return level, flat


def multiplier_to_numeric(level, flat, max_flat=20):
    return (level - 1) * max_flat + flat


def multiplier_to_number(level, flat, max_flat):
    return (level - 1) * max_flat + flat


def rank_multiplier(x):
    parsed = [parse_multiplier(s) for s in x]
    numbered = [multiplier_to_numeric(level, flat) for level, flat in parsed]
    rank = list(np.argsort(np.argsort(numbered)) + 1)
    return rank


def rank_lcr(x):
    x = x.to_list()
    if 'C' not in x:
        mapping = {'L': '1', 'R': '2'}
        return [mapping[s] for s in x]
    elif len(x) == 1:
        return ['3']
    else:
        mapping = {'L': '1', 'C': '2', 'R': '3'}
        return [mapping[s] for s in x]


def rank_a2z(x):

    if 'I' not in x:
        letters = range_letter_upper('A', 'H') + range_letter_upper('J', 'Z')
    else:
        letters = range_letter_upper('A', 'Z')
    mapping = {s: str(i + 1) for i, s in enumerate(letters)}

    return [s if s not in mapping else mapping[s] for s in x]


def rank_g(x):

    mapping = {'G': 0}

    return [s if s not in mapping else mapping[s] for s in x]


def abbr_flat_det(x):
    if 'C' not in x:
        mapping = {
            'LEFT': 'L;1',
            'RIGHT': 'R;2'
        }
    else:
        mapping = {
            'LEFT': 'L;1',
            'RIGHT': 'R;3',
            'CENTRE': 'C;2'
        }

    return [s if s not in mapping else mapping[s] for s in x]


def abbr_level_det(x):
    mapping = {
        'BASEMENT': 'BF',
        'BOTTOM': 'BF',
        'GROUND': 'G;0',
        'FIRST': '1',
        'SECOND': '2',
        'THIRD': '3',
        'FOURTH': '4',
        'MIDDLE': 'M',
        'MID': 'M',
        'ATTIC': '<max>',
        'TOP': '<max>',
    }

    return [s if s not in mapping else mapping[s] for s in x]


# # Calculation for local max


def local_max_det(series):
    all_numbers = []

    for s in series:

        ss = s.split(';')
        for sss in ss:
            if sss.isnumeric():
                all_numbers.append(int(sss))

    if len(all_numbers):
        return max(all_numbers)
    else:
        return 0


sbn_det_dict = {
    'level_det': ['BASEMENT', 'BOTTOM', 'GROUND', 'LOWER', 'FIRST', 'SECOND', 'THIRD',
                  'FOURTH', 'MIDDLE', 'MID', 'UPPER', 'ATTIC', 'TOP'],
    'seperator': ['APARTMENT', 'FLOOR', 'FLAT'],
    'flat_det': ['FRONT', 'REAR', 'LEFT', 'RIGHT', 'CENTRE']
}


expansion_rules = {
    'multiplier': {
        'inclusion': r'^\d+F\d+$',
        'exclusion': r'^\d+$',
        'mapping': rank_multiplier
    },
    'level_det': {
        'inclusion': r'%s' % '|'.join(sbn_det_dict['level_det']),
        'exclusion': r'99999',
        'mapping': abbr_level_det
    },
    'flat_det': {
        'inclusion': r'%s' % '|'.join(sbn_det_dict['flat_det']),
        'exclusion': r'(^d+$)|(^[A-Z]$)',
        'mapping': abbr_flat_det
    },
    'lcr': {
        'inclusion': r'^(L|C|R)$',
        'exclusion': r'(^\d+$)|(^[ABD-KM-QT-Z]$)',
        'mapping': rank_lcr
    },
    'a_to_z': {
        'inclusion': r'^[A-Z]$',
        'exclusion': r'^\d+$',
        'mapping': rank_a2z
    },
    'G': {
        'inclusion': r'^G$',
        'exclusion': r'^[A-FH-Z]$',
        'mapping': rank_g
    }

}


def expand_number_like(group):

    levels = ['number_like_%s' % s for s in range(5)]
    expansion_collection = {k: [] for k in levels}

    expanded = False

    for i in range(1, 5):

        # Start of Level loop

        group_by = levels[:i]

        if len(group_by) == 1:
            group_by = group_by[0]

        for indices, gg in group.groupby(group_by):

            # Start of GroupBy/GG loop

            gg_alt = gg.copy()

            for k, v in expansion_rules.items():

                # Start of Rule Loop

                incl_match = [re.search(r'%s' % v['inclusion'], s) for s in gg[levels[i]]]
                excl_match = [re.search(r'%s' % v['exclusion'], s) for s in gg[levels[i]]]

                incl = [m is not None for m in incl_match]
                excl = [m is not None for m in excl_match]

                if any(incl) and (not any(excl)):

                    mapped = v['mapping'](gg_alt.loc[incl, levels[i]])

                    # print(mapped)

                    gg_alt.loc[incl, levels[i]] = [
                        ';'.join([x, str(y)]) for x, y in zip(gg_alt.loc[incl, levels[i]], mapped)]

                    expanded = True

                # End of Rule Loop

            if any(['<max>' in s for s in gg_alt[levels[i]]]):

                max_det = local_max_det(gg_alt[levels[i]])

                gg_alt[levels[i]] = gg_alt[levels[i]].apply(lambda x: x.replace('<max>', str(max_det + 1)))

            expansion_collection[levels[i]].append(gg_alt[[levels[i]]])

            # End of GroupBy/GG loop

        expansion_collection[levels[i]] = pd.concat(expansion_collection[levels[i]])

        # End of level loop

    if expanded:

        expansion_collection['number_like_0'] = group[['number_like_0']]

        group.loc[:, levels] = pd.concat([df for df in expansion_collection.values()], axis=1)

        return pd.DataFrame.from_records(flatten([expand_number_like_row(row) for _, row in group.iterrows()]))

    else:
        return pd.DataFrame([])


def expand_list(ll):
    return [list(item) for item in itertools.product(*ll)]


def expand_dict(d):
    ll = expand_list(d.values())
    dl = []

    for l in ll:
        dl.append({k: v for k, v in zip(d.keys(), l)})

    return dl


def expand_number_like_row(row):

    levels = ['number_like_%s' % s for s in range(5)]

    row_d = row.to_dict()
    to_expand = {col: list(set(row_d[col].split(';')[1:])) for col in levels if ';' in row_d[col]}

    l_ex = expand_dict(to_expand)

    d_ex = []

    if len(l_ex[0]):
        for d in l_ex:
            row_d.update(d)
            row_d['ex_label'] = row_d['ex_label'] + '_2_nl'
            d_ex.append(row_d)

    return d_ex
