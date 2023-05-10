"""
Overview of Rule parser logics:

- Parse postcode

Exact parsing
- Get info of postcode
- Exact parsing big fields
- Parse numbers

If any field not parsed

Fussy parsing
-
-

Parse text first and then map location

Use text fields for recreation in preprocessing
"""

import re
import json

from src.parser.vocabulary import l_thoroughfare_names, l_posttown_names, l_dl_names, l_ddl_names
from src.utils import span_to_string, repl_char_from_string, span_end


with open('src/parser/pc1_mappings.json', 'r') as f:
    pc1_mappings = json.load(f)

with open('src/parser/pc1_mappings_region.json', 'r') as f:
    pc1_mappings_region = json.load(f)

pc1_mappings['REGION'] = pc1_mappings_region

number_like_regex = {
    'MULTIPLIER': [r'(\d[FKLPS]\d)'],
    'NUMBER_COMPOUND': [r'(\d+[A-Z]?)(?:/|-|\.|\\|,)(\d+[A-Z]?)'],
    'NUMBER_SECONDARY_PREFIX': [r'%s ?(\d+[A-Z]?)' % s
                                for s in ['FLAT', 'F', 'FL', 'BUILDING', 'ROOM', 'BLOCK', 'BONDS?', 'NO']],
    'NUMBER_LIKE': [r'(\d+[A-Z]?)']
}

regex_rules = {
        'POSTCODE': [r'([A-Z]{1,2}[0-9][A-Z0-9]?)(?: +)?([0-9][A-Z]{2})'],
        'POST_TOWN': [r'(?s:.*)(?:\s|,|\n)(%s)(?:\s|$|,|\n)?' % s for s in l_posttown_names],
        'REGION': [r'(?s:.*)(?:\s|,|\n)(%s)(?:\s|$|,|\n)?' % s
                   for s in ['FIFE', 'TAYSIDE', 'LOTHIAN', 'WEST LOTHIAN', 'EAST LOTHIAN', 'MIDLOTHIAN']],
        'THOROUGHFARE': [
                r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % ('\d+', '|'.join(l_thoroughfare_names)),
                r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % ('\d+[A-Z]+', '|'.join(l_thoroughfare_names)),
                r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % ('[A-Z]+\d+', '|'.join(l_thoroughfare_names)),
                r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % ('\d+(?:/|-)\d+', '|'.join(l_thoroughfare_names)),
                r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % ('\d+(?:/|-)[A-Z]+', '|'.join(l_thoroughfare_names)),
                r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % ('[A-Z]+(?:/|-)\d+', '|'.join(l_thoroughfare_names)),
                r'(%s) ((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % ('[A-Z]+(?:/|-)[A-Z]+', '|'.join(l_thoroughfare_names)),
                r'((?:[A-Z]+ ){1,3}(?:%s))(?:\s|$|,|\n)' % '|'.join(l_thoroughfare_names),
                r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % '\d+',
                r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % '\d+[A-Z]+',
                r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % '[A-Z]+\d+',
                r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % '\d+(?:/|-)\d+',
                r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % '\d+(?:/|-)[A-Z]+',
                r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % '[A-Z]+(?:/|-)\d+',
                r'(%s) ((?:[A-Z]+ ){0,2}[A-Z]+)(?:$|,|\n)' % '[A-Z]+(?:/|-)[A-Z]+'
                ],
        'DOUBLE_DEPENDENT_LOCALITY': [r'(?s:.*)(?:\s|^|,|\n)(%s)(?:\s|$|,|\n)?' % s for s in l_ddl_names],
        'DEPENDENT_LOCALITY': [r'(?s:.*)(?:\s|^|,|\n)(%s)(?:\s|$|,|\n)?' % s for s in l_dl_names]
}


def _pattern_mapping(l_patterns, string, keep='longest'):

    l_match = []
    for p in l_patterns:
        l_match.extend(list(re.finditer(p, string)))

    if len(l_match):

        if keep == 'first':
            return [l_match[0]]

        elif keep == 'all':
            return l_match

        elif keep == 'longest':
            return [max(l_match, key=lambda x: len(x.group(0)))]

    else:
        return []


def exact_parsing(address):

    exact_parsing_fields = ['REGION', 'THOROUGHFARE', 'DOUBLE_DEPENDENT_LOCALITY', 'DEPENDENT_LOCALITY', 'POST_TOWN']

    string = {}
    span = {}

    need_fuzzy_parsing = False
    residual_address = address

    try:

        match = list(re.finditer(regex_rules['POSTCODE'][0], address))[0]

        string['POSTCODE'] = match.group(0)
        string['PC1'] = match.group(1)
        string['PC2'] = match.group(2)

        span['POSTCODE'] = match.span(0)

        for field in exact_parsing_fields:

            if string['PC1'] in pc1_mappings[field]:
                l_patterns = pc1_mappings[field][string['PC1']]
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

    return {'STRING': string, 'SPAN': span, 'need_fuzzy_parsing': need_fuzzy_parsing, 'residual_address': residual_address}


def number_like_parsing(address):

    residual_address = address

    string = {}
    span = {}

    l_match = _pattern_mapping(number_like_regex['MULTIPLIER'], address, keep='first')

    if len(l_match):
        match = l_match[0]

        string['SECONDARY_NUMBER'] = match.group(1)
        span['SECONDARY_NUMBER'] = match.span(1)

        residual_address = repl_char_from_string(residual_address, span['SECONDARY_NUMBER'])

        match = re.search(number_like_regex['NUMBER_LIKE'][0], residual_address)

        if match is not None:

            string['PRIMARY_NUMBER'] = match.group(1)
            span['PRIMARY_NUMBER'] = match.span(1)

            residual_address = repl_char_from_string(residual_address, span['PRIMARY_NUMBER'])

        return {'STRING': string, 'SPAN': span, 'NUMBER_TYPE': 'MULTIPLIER', 'residual_address': residual_address}

    l_match = _pattern_mapping(number_like_regex['NUMBER_COMPOUND'], address, keep='first')

    if len(l_match):

        match = l_match[0]

        string['PRIMARY_NUMBER'] = match.group(1)
        span['PRIMARY_NUMBER'] = match.span(1)

        string['SECONDARY_NUMBER'] = match.group(2)
        span['SECONDARY_NUMBER'] = match.span(2)

        residual_address = repl_char_from_string(residual_address, match.span(0))

        return {'STRING': string, 'SPAN': span, 'NUMBER_TYPE': 'NUMBER_COMPOUND', 'residual_address': residual_address}

    l_match = _pattern_mapping(number_like_regex['NUMBER_SECONDARY_PREFIX'], address)

    if len(l_match):
        match = l_match[0]

        string['SECONDARY_NUMBER'] = match.group(1)
        span['SECONDARY_NUMBER'] = match.span(1)

        residual_address = repl_char_from_string(residual_address, match.span(0))

        match = re.search(number_like_regex['NUMBER_LIKE'][0], residual_address)

        if match is not None:
            string['PRIMARY_NUMBER'] = match.group(1)
            span['PRIMARY_NUMBER'] = match.span(1)

            residual_address = repl_char_from_string(residual_address, match.span(0))

        return {'STRING': string, 'SPAN': span, 'NUMBER_TYPE': 'NUMBER_SECONDARY_PREFIX', 'residual_address': residual_address}

    l_match = _pattern_mapping(number_like_regex['NUMBER_LIKE'], address, keep='all')

    if len(l_match):
        string['PRIMARY_NUMBER'] = l_match[-1].group(1)
        span['PRIMARY_NUMBER'] = l_match[-1].span(1)

        residual_address = repl_char_from_string(residual_address, l_match[-1].span(1))

        if len(l_match) >= 2:

            string['SECONDARY_NUMBER'] = l_match[-2].group(1)
            span['SECONDARY_NUMBER'] = l_match[-2].span(1)

            residual_address = repl_char_from_string(residual_address, l_match[-2].span(1))

        return {'STRING': string, 'SPAN': span, 'NUMBER_TYPE': 'NUMBER_LIKE', 'residual_address': residual_address}

    return {'STRING': {'PRIMARY_NUMBER': '', 'SECONDARY_NUMBER': ''},
            'SPAN': {'PRIMARY_NUMBER': None, 'SECONDARY_NUMBER': None},
            'NUMBER_TYPE': 'NO_NUMBERS',
            'residual_address': residual_address}


def fuzzy_parsing(address):

    span = {}
    string = {}

    residual_address = address

    for key, l_regex in regex_rules.items():

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
                    string[key] = match.group(0)
                    string['PC1'] = match.group(1)
                    string['PC2'] = match.group(2)

                    residual_address = repl_char_from_string(residual_address, match.span(0))
                else:
                    span[key] = match.span(1)
                    string[key] = match.group(1)

                    residual_address = repl_char_from_string(residual_address, match.span(1))

    return {'STRING': string, 'SPAN': span, 'residual_address': residual_address}


def rule_parsing(address):

    string = {}
    span = {}

    parsed = exact_parsing(address)

    if parsed['need_fuzzy_parsing']:
        parsed = fuzzy_parsing(address)

    parsed_number_like = number_like_parsing(parsed['residual_address'])

    string.update(parsed['STRING'])
    span.update(parsed['SPAN'])

    string.update(parsed_number_like['STRING'])
    span.update(parsed_number_like['SPAN'])

    residual_address = parsed_number_like['residual_address']

    return {'STRING': string, 'SPAN': span, 'residual_address': residual_address,
            'NUMBER_TYPE': parsed_number_like['NUMBER_TYPE']}

