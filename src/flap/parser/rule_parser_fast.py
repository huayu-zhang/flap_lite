import re
import json
from tqdm.contrib.concurrent import process_map
import pandas as pd

from copy import deepcopy

from flap.utils import span_to_string, repl_char_from_string, span_end, flatten
from flap.database.sql import SqlDB
from flap.preprocessing import address_line_preproc
from flap.utils import available_cpu_count


class RuleParserFast:

    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            loaded = json.load(f)
        return loaded

    def __init__(self, sql_db=None):

        self.l_thoroughfare_names = None
        self.l_posttown_names = None
        self.pc1_mappings = None
        self.pc1_mappings_region = None

        self.regex_rules = None
        self.need_fuzzy_parsing = False

        if sql_db is not None:
            self.l_thoroughfare_names = self.load_json(sql_db.sub_paths['thoroughfare_patterns'])
            self.l_posttown_names = self.load_json(sql_db.sub_paths['unique_POST_TOWN'])
            self.pc1_mappings = self.load_json(sql_db.sub_paths['pc1_mappings'])
            self.pc1_mappings_region = self.load_json(sql_db.sub_paths['pc1_mappings_region'])

            self.pc1_mappings['REGION'] = self.pc1_mappings_region

            self.regex_rules = {
                'POSTCODE': [r'([A-Z]{1,2}[0-9][A-Z0-9]?)(?: *\n?)?([0-9][A-Z]{2})'],
                'POST_TOWN': [r'(?s:.*)(?:\s|,|\n)(%s)(?:\s|$|,|\n)?' % s for s in self.l_posttown_names],
                'REGION': [r'(?s:.*)(?:\s|,|\n)(%s)(?:\s|$|,|\n)?' % s
                           for s in ['FIFE', 'TAYSIDE', 'LOTHIAN', 'WEST LOTHIAN', 'EAST LOTHIAN', 'MIDLOTHIAN']],
                'THOROUGHFARE': [
                    r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % (r'\d+', '|'.join(self.l_thoroughfare_names)),
                    r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % (r'\d+[A-Z]+', '|'.join(self.l_thoroughfare_names)),
                    r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % (r'[A-Z]+\d+', '|'.join(self.l_thoroughfare_names)),
                    r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % (r'\d+(?:/|-)\d+', '|'.join(self.l_thoroughfare_names)),
                    r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % (r'\d+(?:/|-)[A-Z]+', '|'.join(self.l_thoroughfare_names)),
                    r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % (r'[A-Z]+(?:/|-)\d+', '|'.join(self.l_thoroughfare_names)),
                    r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % (r'[A-Z]+(?:/|-)[A-Z]+', '|'.join(self.l_thoroughfare_names)),
                    r'((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % '|'.join(self.l_thoroughfare_names),
                    r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % r'\d+',
                    r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % r'\d+[A-Z]+',
                    r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % r'[A-Z]+\d+',
                    r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % r'\d+(?:/|-)\d+',
                    r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % r'\d+(?:/|-)[A-Z]+',
                    r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % r'[A-Z]+(?:/|-)\d+',
                    r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % r'[A-Z]+(?:/|-)[A-Z]+'
                ]
            }

    def copy(self):
        return deepcopy(self)

    def exact_parsing(self, address):

        exact_parsing_fields = ['REGION', 'THOROUGHFARE', 'DOUBLE_DEPENDENT_LOCALITY', 'DEPENDENT_LOCALITY', 'POST_TOWN']

        string = {}
        span = {}

        need_fuzzy_parsing = False
        residual_address = address

        try:

            match = list(re.finditer(self.regex_rules['POSTCODE'][0], address))[0]

            string['POSTCODE'] = ' '.join([match.group(1), match.group(2)])
            string['PC1'] = match.group(1)
            string['PC2'] = match.group(2)

            span['POSTCODE'] = match.span(0)

            for field in exact_parsing_fields:

                if string['PC1'] in self.pc1_mappings[field]:
                    l_patterns = self.pc1_mappings[field][string['PC1']]
                    l_regex = [r'(?s:.*)(%s)' % s for s in l_patterns]

                    match_list = []

                    for regex in l_regex:
                        match = re.search(pattern=regex, string=address)
                        if match is not None:
                            match_list.append(match)

                    if len(match_list):
                        match = max(match_list, key=lambda x: len(x.group(1)))
                        string[field] = match.group(1)
                        span[field] = match.span(1)

            for key in ['THOROUGHFARE', 'POST_TOWN']:
                if key not in string:
                    need_fuzzy_parsing = True

            for sp in span.values():
                residual_address = repl_char_from_string(residual_address, sp)

        except IndexError:
            need_fuzzy_parsing = True

        string['residual_address'] = residual_address

        return string, need_fuzzy_parsing

    def fuzzy_parsing(self, address):

        span = {}
        string = {}

        residual_address = address

        for key, l_regex in self.regex_rules.items():

            match_type = 0

            if key == 'THOROUGHFARE':

                l_match_results = []

                for regex in l_regex:

                    for i in range(2):
                        match = re.search(pattern=regex, string=residual_address)

                        if match:

                            l_match_results.append(
                                {'match': match,
                                 'match_type': match_type}
                                                  )

                            if len(match.groups()) == 1:

                                residual_address = repl_char_from_string(residual_address, match.span(1))

                            elif len(match.groups()) == 2:

                                match_start, match_end = match.span(2)

                                residual_address = repl_char_from_string(residual_address, (match_start, match_end))

                    if len(l_match_results) >= 2:

                        break

                    match_type += 1

                l_match_results = sorted(l_match_results, key=lambda x: span_end(x['match'].span()), reverse=True)

                for i, match_result in enumerate(l_match_results):

                    if i == 0:
                        field_name = 'THOROUGHFARE'
                    elif i == 1:
                        field_name = 'THOROUGHFARE_EXTRA'
                    else:
                        field_name = 'THOROUGHFARE'

                    if len(match_result['match'].groups()) == 1:

                        span[field_name] = match_result['match'].span(1)
                        string[field_name] = match_result['match'].group(1)

                        residual_address = repl_char_from_string(residual_address, match_result['match'].span(1))

                    elif len(match_result['match'].groups()) == 2:

                        span[field_name] = match_result['match'].span(2)
                        string[field_name] = match_result['match'].group(2)

                        residual_address = repl_char_from_string(residual_address, match_result['match'].span(2))

            else:

                match_list = []

                for regex in l_regex:

                    match = re.search(pattern=regex, string=residual_address)

                    if match is not None:
                        match_list.append(match)

                if len(match_list):

                    match = max(match_list, key=lambda x: len(x.group(1)))

                    if key == 'POSTCODE':
                        span[key] = match.span(0)
                        string[key] = ' '.join([match.group(1), match.group(2)])
                        string['PC1'] = match.group(1)
                        string['PC2'] = match.group(2)

                        residual_address = repl_char_from_string(residual_address, match.span(0))
                    else:
                        span[key] = match.span(1)
                        string[key] = match.group(1)

                        residual_address = repl_char_from_string(residual_address, match.span(1))

        string['residual_address'] = residual_address

        return string

    def parse(self, address, method='fast'):

        parsed = {'STRING': {}, 'FOR_QUERY': {}}

        if method == 'fast':

            parsed.update(fast_parsing(address))
            parsed['FOR_QUERY'].update(parsed['NUMBER_LIKE'])
            parsed['FOR_QUERY']['TEXTUAL'] = parsed['TEXTUAL']
            try:
                parsed['FOR_QUERY']['POSTCODE'] = parsed['POSTCODE']
                parsed['FOR_QUERY'].update(parsed['POSTCODE_SPLIT'])
            except KeyError:
                pass

            return parsed

        elif method == 'all':

            parsed.update(fast_parsing(address))
            parsed['FOR_QUERY'].update(parsed['NUMBER_LIKE'])
            parsed['FOR_QUERY']['TEXTUAL'] = parsed['TEXTUAL']
            try:
                parsed['FOR_QUERY']['POSTCODE'] = parsed['POSTCODE']
                parsed['FOR_QUERY'].update(parsed['POSTCODE_SPLIT'])
            except KeyError:
                pass

            string, need_fuzzy_parsing = self.exact_parsing(address)
            parsed['STRING'].update(string)
            parsed['FOR_QUERY'].update(string)

            if need_fuzzy_parsing:
                string = self.fuzzy_parsing(address)
                parsed['STRING'].update(string)
                parsed['FOR_QUERY'].update(string)

        else:
            raise ValueError('`method` must be `fast` or `all`')

        return parsed

    def parse_one_input_row(self, iterrow):

        _, row = iterrow

        record = row.to_dict()

        try:
            address, _ = address_line_preproc(row['input_address'])

            record['preproc_address'] = address

            record.update(self.parse(address, method='all')['FOR_QUERY'])

        except:
            pass

        return record

    def parse_batch_in_df(self, df, max_workers=None, chunk_size=None):

        if max_workers is None:
            max_workers = available_cpu_count()

        if chunk_size is None:
            chunk_size = int(len(df) / max_workers) + int((len(df) % max_workers) > 0)

        records = process_map(self.parse_one_input_row, df.iterrows(),
                              max_workers=max_workers, chunksize=chunk_size, total=len(df))

        return pd.DataFrame.from_records(records)


NUMBER_LIKE_MASTER_REGEX = re.compile(
    r"(FLAT|UNIT|HOUSE|BUILDING|ROOM|BLOCK|BONDS?|FL|PF|BF|GF|APARTMENT|\(F|F|-|\()? ?"
    r"((\dF\d+)|(\d+[A-Z]\d+)|([A-EG-Z])?(\d+)([A-Z])?|(?<!')(^|\b)([A-Z]|GROUND)($|\b)(?!'))"
    r"\)?")

POSTCODE_MASTER_REGEX = re.compile(r'([A-Z]{1,2})([0-9][A-Z0-9]?)(?: *\n?)?([0-9])([A-Z])([A-Z])')

SPLIT_MASTER_REGEX = re.compile(r"(?<=\d)(?=\D)|(?=\d)(?<=\D)")


def fast_parsing(address):

    p = NUMBER_LIKE_MASTER_REGEX

    split_pattern = SPLIT_MASTER_REGEX

    parsed = {}

    p_pc = POSTCODE_MASTER_REGEX

    match_pc = re.search(p_pc, address)

    string_pc = {}

    if match_pc is not None:

        string_pc['pc_area'], string_pc['pc_district'], string_pc['pc_sector'], string_pc['pc_unit_0'], string_pc['pc_unit_1'] = \
            [match_pc.group(i) for i in range(1, 6)]

        postcode = ' '.join([''.join([string_pc['pc_area'], string_pc['pc_district']]),
                             ''.join([string_pc['pc_sector'], string_pc['pc_unit_0'], string_pc['pc_unit_1']])])

        parsed['POSTCODE_SPLIT'] = string_pc
        parsed['POSTCODE'] = postcode

        address = repl_char_from_string(address, match_pc.span())

    matches = list(re.finditer(p, address))

    prefix = [(match.group(1) is not None) or (re.search(r'\dF\d', match.group(2)) is not None) for match in matches]

    units = [re.split(split_pattern, match.group(2))
             if re.search(r'\dF\d', match.group(2)) is None else [match.group(2)]
             for match in matches]

    units_tmp = units.copy()

    # Fix units

    for i in range(len(matches) - 1):
        if (matches[i+1].span(2)[0] - matches[i].span(2)[1]) == 1:
            char = address[matches[i].span(2)[1]:matches[i+1].span(2)[0]]
            if char != ' ':
                units[i + 1] = units_tmp[i] + units_tmp[i + 1]
                units[i] = []
                units_tmp = units.copy()

                prefix[i] = True
                prefix[i + 1] = True

    for i in range(len(matches) - 1):
        if (not prefix[i]) and prefix[i + 1]:
            p = i + 2
            while (p < len(matches)) and (prefix[p]):
                p += 1
            p -= 1

            units[i], units[p] = units[p], units[i]

    res = flatten(units[::-1])

    if len(res) < 5:
        res += [''] * (5 - len(res))
    elif len(res) > 5:
        res = res[:5]

    for match in matches:
        address = repl_char_from_string(address, match.span())

    string_number_like = {'number_like_%s' % i: res[i] for i in range(len(res))}

    parsed['NUMBER_LIKE'] = string_number_like
    parsed['TEXTUAL'] = address

    return parsed


# import time
#
#
# sql_db = SqlDB('/home/huayu_ssh/PycharmProjects/dres_r/db/scotland_curl')
# parser = RuleParserFast(sql_db)
#
# address = """7 F3 GRANVILLE TERRACE, EDINBURGH"""

# Exact parsing
#
# t0 = time.time()
#
# for i in range(100):
# parsed = parser.parse(address, method='all')
#
# parser.fuzzy_parsing(address)
#
# print(time.time() - t0)
#
# fast_parsing(address)
#
# parser.exact_parsing(address)
# parser.fuzzy_parsing(address)
#

# print(parsed)
# t0 = time.time()
#
# for i in range(100):
#     parsed = parser.parse(address, method='all')
#
# print(time.time() - t0)
# print(parsed)

# r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % ('\d+', '|'.join(parser.l_thoroughfare_names))

#
# # Fussy parsing
#
# t0 = time.time()
#
# for i in range(10):
#     parser.fuzzy_parsing(address)
#
# print(time.time() - t0)
#
#
# t0 = time.time()
#
# for i in range(100):
#
#     for p in parser.regex_rules['POST_TOWN']:
#         re.search(p, address)
#
# print(time.time() - t0)
#
#
# t0 = time.time()
#
# for i in range(100):
#
#     for p in parser.l_posttown_names:
#         p in address
#
# print(time.time() - t0)
#
# t0 = time.time()
#
# for i in range(1000):
#
#     fast_parsing(address)
#
# print(time.time() - t0)
#
#

#
#
# matches = re.finditer(NUMBER_LIKE_MASTER_REGEX, address)
#
# for match in matches:
#     print(match.span())
#     print(match.group())
#     print(match.groups())
#
# parse_number_like(address)
#



