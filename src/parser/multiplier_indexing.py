# """
# build index for data
#
# index = '[PRIMARY_NUMBER]-[PC]'
#
# build index for database
#
# index = '[BUILDING_NUMBER]-[PC]'
#
# second_index = '[BUILDING_NUMBER]-[PT]-[THO]'
#
# """
#
#
# import pandas as pd
# import re
#
# from src.parser.rule_parser import rule_parsing
# from tqdm.contrib.concurrent import process_map
# from tqdm import tqdm
#
#
# def parse_multiplier(s):
#
#     match = re.search(r'(\d)[FKLPS](\d)', s)
#
#     if match is not None:
#
#         return int(match.group(1)), int(match.group(2))
#
#     else:
#         return None
#
#
# def local_multiplier_max(list_of_multipliers):
#
#     parsed_list = [parse_multiplier(s) for s in list_of_multipliers if parse_multiplier(s) is not None]
#
#     try:
#
#         max_level = max([x[0] for x in parsed_list])
#         max_flat = max(x[1] for x in parsed_list)
#
#         return {'max_level': max_level,
#                 'max_flat': max_flat}
#
#     except ValueError:
#         return None
#
#
# def guess_multiplier(index, local_max, local_n_subs):
#
#     if (index in local_max) and (index in local_n_subs):
#
#         max_level = local_max[index]['max_level']
#         max_flat = local_max[index]['max_flat']
#         n_sub = local_n_subs[index]
#
#         if max_flat > 1:
#
#             if max_level * max_flat == n_sub:
#                 return {'multiplier': max_flat, 'confidence': 'exact'}
#
#             elif n_sub % max_flat == 0:
#                 return {'multiplier': max_flat, 'confidence': 'high'}
#
#     elif index in local_n_subs:
#
#         n_sub = local_n_subs[index]
#
#         if (n_sub % 2 == 0) and (n_sub % 3 == 0):
#
#             return {'multiplier': 2, 'confidence': 'low'}
#
#         elif n_sub % 2 == 0:
#
#             return {'multiplier': 2, 'confidence': 'medium'}
#
#         elif n_sub % 3 == 0:
#
#             return {'multiplier': 3, 'confidence': 'medium'}
#
#     elif index in local_max:
#
#         max_flat = local_max[index]['max_flat']
#
#         if max_flat > 1:
#
#             return {'multiplier': max_flat, 'confidence': 'low'}
#
#     return {'multiplier': 2, 'confidence': 'none'}
#
#
# def index_multiplier(address):
#     try:
#         MULTIPLIER_REGEX = r'(\d[FKLPSfklps]\d)'
#
#         if re.search(MULTIPLIER_REGEX, address) is not None:
#             parsed = rule_parsing(address)
#
#             return {
#                         'INDEX': '-'.join([parsed['STRING']['PRIMARY_NUMBER'],
#                                            parsed['STRING']['POSTCODE']]),
#                         'SECONDARY_NUMBER': parsed['STRING']['SECONDARY_NUMBER']
#                     }
#
#         else:
#             return None
#
#     except:
#         return None
#
#
# def build_index_for_data(data):
#     multiplier_collection = {}
#
#     indexed_list = process_map(index_multiplier, data['input_address'].tolist(), chunksize=1000)
#
#     for indexed in tqdm(indexed_list):
#
#         if indexed is not None:
#             if indexed['INDEX'] in multiplier_collection:
#                 multiplier_collection[indexed['INDEX']].append(indexed['SECONDARY_NUMBER'])
#             else:
#                 multiplier_collection[indexed['INDEX']] = [indexed['SECONDARY_NUMBER']]
#
#     return multiplier_collection
#
#
# def yield_batch(batch_size, config):
#
#     sql_header = {
#         "RECORD_IDENTIFIER": "INTEGER",
#         "CHANGE_TYPE": "TEXT",
#         "PRO_ORDER": "TEXT",
#         "UPRN": "TEXT",
#         "UDPRN": "TEXT",
#         "ORGANISATION_NAME": "TEXT",
#         "DEPARTMENT_NAME": "TEXT",
#         "SUB_BUILDING_NAME": "TEXT",
#         "BUILDING_NAME": "TEXT",
#         "BUILDING_NUMBER": "TEXT",
#         "DEPENDENT_THOROUGHFARE": "TEXT",
#         "THOROUGHFARE": "TEXT",
#         "DOUBLE_DEPENDENT_LOCALITY": "TEXT",
#         "DEPENDENT_LOCALITY": "TEXT",
#         "POST_TOWN": "TEXT",
#         "POSTCODE": "TEXT",
#         "POSTCODE_TYPE": "TEXT",
#         "DELIVERY_POINT_SUFFIX": "TEXT",
#         "WELSH_DEPENDENT_THOROUGHFARE": "TEXT",
#         "WELSH_THOROUGHFARE": "TEXT",
#         "WELSH_DOUBLE_DEPENDENT_LOCALITY": "TEXT",
#         "WELSH_DEPENDENT_LOCALITY": "TEXT",
#         "WELSH_POST_TOWN": "TEXT",
#         "PO_BOX_NUMBER": "TEXT",
#         "PROCESS_DATE": "TEXT",
#         "START_DATE": "TEXT",
#         "END_DATE": "TEXT",
#         "LAST_UPDATE_DATE": "TEXT",
#         "ENTRY_DATE": "TEXT",
#         "ex_label": "TEXT",
#         "ex_uprn": "TEXT"
#     }
#
#     DB_PATH = 'db/sql/ADDRESSPREMIUM_EX.sqlite3'
#     COLUMNS = list(sql_header.keys())
#
#     con = sqlite3.connect(DB_PATH)
#     cur = con.cursor()
#     cur.execute(config['SQL'])
#
#     res = cur.fetchmany(batch_size)
#     next_res = cur.fetchone()
#
#     while len(res):
#
#         try:
#             while config['match_func'](next_res, res):
#                 res.append(next_res)
#                 next_res = cur.fetchone()
#
#         except TypeError:
#             pass
#
#         res_df = pd.DataFrame.from_records(res)
#         res_df.columns = COLUMNS
#
#         yield res_df
#
#         if next_res is None:
#             break
#         else:
#             res = [next_res]
#             res.extend(cur.fetchmany(batch_size))
#             next_res = cur.fetchone()
#
#     con.close()
#
#
# def index_local_neighbourhood_database(batch):
#
#     collections = []
#
#     for (bnu, pc, ex_label), group in batch.groupby(['BUILDING_NUMBER', 'POSTCODE', 'ex_label']):
#
#         if 'syn' not in ex_label:
#             if len(bnu):
#                 if len(group) > 1:
#                     collections.append({
#                         'index': '-'.join([bnu, pc]),
#                         'number_of_subs': len(group)
#                     })
#     return collections
#
#
# def build_index_for_database():
#     import sqlite3
#     import itertools
#
#     TABLE_NAME = 'Record_28_Expanded'
#
#     BATCH_SIZE = int(1e5)
#
#     batch_configs = {
#         'pc': {
#             'SQL': """
#                 SELECT *
#                 FROM %s
#                 ORDER BY POSTCODE
#             """ % TABLE_NAME,
#             'match_func': lambda next_res, res: next_res[15] == res[-1][15]
#         }
#     }
#
#     batch_gen = yield_batch(BATCH_SIZE, batch_configs['pc'])
#
#     list_local_index_database = process_map(index_local_neighbourhood_database, batch_gen)
#
#     list_local_index_database_flattened = list(itertools.chain(*list_local_index_database))
#
#     local_index_database = {d['index']: d['number_of_subs'] for d in list_local_index_database_flattened}
#
#     return local_index_database
#
# #
# # # local_n_subs = build_index_for_database()
# # #
# # with open('/home/huayu_ssh/PycharmProjects/dres_r/db/local_n_subs.json', 'r') as f:
# #     local_n_subs = json.load(f)
# #
# #
# # data = pd.read_csv('/home/huayu_ssh/PycharmProjects/dres_r/projects/dev_1103/input/input_df.csv', index_col=0)
# #
# # index_data = build_index_for_data(data)
# #
# # local_max = {index: local_multiplier_max(list_of_multipliers) for index, list_of_multipliers in index_data.items()
# #              if local_multiplier_max(list_of_multipliers) is not None}
# #
# #
# # import json
# # #
# # #
# # # with open('db/local_n_subs.json', 'w') as f:
# # #     json.dump(local_n_subs, f, indent=4)
# # #
# # #
# # with open('projects/dev_1103/local_max.json', 'w') as f:
# #     json.dump(local_max, f, indent=4)
# #
# #
# # indexed_multipliers = {index: guess_multiplier(index) for index in index_data}
# #
# # with open('projects/dev_1103/indexed_multipliers.json', 'w') as f:
# #     json.dump(indexed_multipliers, f, indent=4)
