"""
Input fields

input_id, input_address, flap_score, flap_uprn, label_uprn optionals ['uprn_row', 'error']

Output

Evaluation metrics

Per strata-value, micro metrics

Per strata, macro metrics

Average of per strata, final metrics


The process for eval_by_sampling_for_val

<input> = ['input_id', 'input_address', OPTIONAL: 'uprn']
<sampling> = ['strata', 'strata_value', 'sample_weight']
<verdict> = ['linking_verdict', 'clf_verdict']
<metrics> = ['accuracy', 'accuracy_strict', 'precision', 'recall', 'f1']
<linking_verdict> = ['FN-c_ps_npr', 'FP_Not_Possible-nc_nps_pr',
       'FP_Wrong_One-nc_ps_pr', 'Good-TN-nc_nps_npr', 'Good-TP-c_ps_pr',
       'TN-Not_found-nc_ps_npr']
<clf_verdict> = ['clf_FN', 'clf_FP', 'clf_TN', 'clf_TP']

- Sampling the input addresses
    - Match and Parce: data[<input>]
        --> data[<input>, <matching>, <parsing>]
    - Sample:
        --> sampled[<input>, <matching>, <sampling>]

- Review and Add true UPRN
    - sampled[<input>, <matching>, <sampling>] --> sampled_with_review[..., 'review']
    review should take 3 kinds of values
    'y' is correct
    'n' is not possible
    regex'\\d+' or numeric string to provide the correct UPRN

- Summarise
    - Add verdict
        sampled_with_review[<input>, <matching>, <sampling>]
        --> sampled_with_review[..., 'linking_verdict', 'clf_verdict']
    - Summarise performance
        sampled_with_review --> performance['name_of_group', <linking_verdicts>, <clf_verdicts>, <metrics>]


Mode `val` validation
Sampling --> Manual Review --> Summarise

Mode `dev` development
Sampling --> Add Review --> Summarise

"""

import warnings
import re
import os

import pandas as pd
from flap.parser.rule_parser_fast import RuleParserFast
from flap.database.sql import SqlDB
from flap.api.score_and_match import score_and_match


def length_number_like(row):
    return sum([row['number_like_%s' % i] != '' for i in range(5)])


def prepare_input_for_sampling(input_csv, db_path, spd_path=None, classifier_model_path=None,
                               random_sample_size=10000, random_state=666):

    if isinstance(input_csv, str):
        data = pd.read_csv(input_csv, dtype='object', usecols=['input_id', 'input_address', 'uprn']).sample(
            random_sample_size, random_state=random_state)

    elif isinstance(input_csv, pd.DataFrame):
        data = input_csv[['input_id', 'input_address', 'uprn']].sample(
            random_sample_size, random_state=random_state).copy()
    else:
        raise TypeError('input_csv should be path(str) or pd.DataFrame')

    # Match

    if 'flap_match_score' not in data:
        match_results = score_and_match(data, db_path, classifier_model_path=classifier_model_path,
                                        score_batch_size=random_sample_size,
                                        match_batch_size=int(random_sample_size/20))

        data = pd.merge(left=data, right=match_results[['input_id', 'flap_match_score', 'flap_uprn', 'uprn_row']],
                        on='input_id', how='left')

    data['flap_score_decile'] = (data.flap_match_score.astype(float) * 10 - 1e-8).astype(int)

    # Parse
    sql_db = SqlDB(db_path)

    parser = RuleParserFast(sql_db)

    if 'number_like_0' not in data:
        data = parser.parse_batch_in_df(data)

    p_multi = re.compile(r'\d[A-Z]\d+')
    data['has_multiplier'] = data.input_address.apply(lambda x: re.search(p_multi, x) is not None)

    data['length_number_like_parsing'] = data.apply(length_number_like, axis=1)

    if spd_path is not None:

        spd = pd.read_csv(spd_path, dtype='object',
                          usecols=['Postcode', 'UrbanRural8Fold2020Code', 'ScottishIndexOfMultipleDeprivation2020Rank'])

        spd_name_mapper = {
            'Postcode': 'postcode_spd',
            'UrbanRural8Fold2020Code': 'ur_8fold',
            'ScottishIndexOfMultipleDeprivation2020Rank': 'simd_2020'
        }

        spd = spd.rename(mapper=spd_name_mapper, axis=1)

        spd = spd[~spd.postcode_spd.duplicated()]

        data = pd.merge(left=data, right=spd, left_on='POSTCODE', right_on='postcode_spd', how='left')

    return data


def stratified_sampling_by_column_value(data, **stratify_config):
    sampled_groups = []

    for col in stratify_config:
        try:
            for i, group in data.groupby(col, dropna=False):
                n = min(stratify_config[col], len(group))

                sampled_group = group[['input_id', 'input_address', 'uprn', 'uprn_row',
                                       'flap_match_score', 'flap_uprn']].sample(n)
                sampled_group['strata'] = col
                sampled_group['strata_value'] = i
                sampled_group['sample_weight'] = len(group) / len(data) / n
                sampled_group['review'] = ''

                sampled_groups.append(sampled_group)

        except KeyError:

            warnings.warn('Columns %s does not exist, skipped for sampling' % col)

    sampled = pd.concat(sampled_groups)

    return sampled


def stratified_sampling_for_review(input_csv, db_path, spd_path=None,
                                   random_sample_size=10000, random_state=666, **stratify_config):
    data = prepare_input_for_sampling(input_csv=input_csv, db_path=db_path, spd_path=spd_path,
                                      random_sample_size=random_sample_size, random_state=random_state)

    sampled = stratified_sampling_by_column_value(data, **stratify_config)

    return sampled


def uprn_in_db(uprn, sql_db):
    query = 'select UPRN from indexed where UPRN=="%s"' % uprn

    return len(sql_db.sql_query(query=query)) > 0


def add_review(sampled, sql_db):
    sampled['review'] = sampled.apply(
        lambda row: 'y' if row['uprn'] == row['flap_uprn'] else 'n'
        if not uprn_in_db(row['uprn'], sql_db) else row['uprn'], axis=1)

    sampled_with_review = sampled

    return sampled_with_review


def _add_linking_verdict_to_reviewed_samples(row):
    correct = True if row['review'] == 'y' else False
    possible = False if row['review'] == 'n' else True
    y_pred = True if row['flap_match_score'] > 0.5 else False

    if correct and possible and y_pred:
        return 'Good-TP-c_ps_pr'

    if correct and possible and not y_pred:
        return 'FN-c_ps_npr'

    if not correct and possible and y_pred:
        return 'FP_Wrong_One-nc_ps_pr'

    if not correct and not possible and y_pred:
        return 'FP_Not_Possible-nc_nps_pr'

    if not correct and possible and not y_pred:
        return 'TN-Not_found-nc_ps_npr'

    if not correct and not possible and not y_pred:
        return 'Good-TN-nc_nps_npr'


def _add_classifier_verdict_to_reviewed_samples(row):
    correct = True if row['review'] == 'y' else False
    y_pred = True if row['flap_match_score'] > 0.5 else False

    if correct and y_pred:
        return 'clf_TP'

    if correct and not y_pred:
        return 'clf_FN'

    if not correct and y_pred:
        return 'clf_FP'

    if not correct and not y_pred:
        return 'clf_TN'


def add_verdict(sampled_with_review):
    sampled_with_review['link_verdict'] = sampled_with_review.apply(_add_linking_verdict_to_reviewed_samples, axis=1)
    sampled_with_review['clf_verdict'] = sampled_with_review.apply(_add_classifier_verdict_to_reviewed_samples, axis=1)

    return sampled_with_review


def summarise_performance_from_verdict(group, add_name=None):
    record = {
        'name_of_group': add_name,
        'FN-c_ps_npr': 0,
        'FP_Not_Possible-nc_nps_pr': 0,
        'FP_Wrong_One-nc_ps_pr': 0,
        'Good-TN-nc_nps_npr': 0,
        'Good-TP-c_ps_pr': 0,
        'TN-Not_found-nc_ps_npr': 0,
        'clf_FN': 0,
        'clf_FP': 0,
        'clf_TN': 0,
        'clf_TP': 0,
        'accuracy': 0,
        'accuracy_strict': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }

    record.update(group.groupby('link_verdict')['sample_weight'].sum().to_dict())
    record.update(group.groupby('clf_verdict')['sample_weight'].sum().to_dict())

    record['accuracy'] = record['FN-c_ps_npr'] + record['Good-TN-nc_nps_npr'] + record['Good-TP-c_ps_pr']
    record['accuracy_strict'] = record['Good-TN-nc_nps_npr'] + record['Good-TP-c_ps_pr']

    try:
        record['precision'] = record['clf_TP'] / (record['clf_TP'] + record['clf_FP'])
        record['recall'] = record['clf_TP'] / (record['clf_TP'] + record['clf_FN'])
        record['f1'] = 2 * record['clf_TP'] / (2 * record['clf_TP'] + record['clf_FP'] + + record['clf_FN'])
    except ZeroDivisionError:
        pass

    return record


def summarise_performance_in_stratified_sampled(sampled_for_review):
    records = []

    for i, group in sampled_for_review.groupby('strata'):
        records.append(summarise_performance_from_verdict(group, add_name=i))

    performance = pd.DataFrame.from_records(records)

    average_row = performance.iloc[:, 1:].apply(lambda x: sum(x) / len(x)).to_dict()

    average_row['name_of_group'] = 'average'

    records.append(average_row)

    for i, group in sampled_for_review.groupby('strata'):

        if group.strata_value.nunique() <= 10:

            for ii, gg in group.groupby('strata_value'):
                records.append(summarise_performance_from_verdict(gg, add_name='%s-%s' % (i, ii)))

    performance = pd.DataFrame.from_records(records)

    return performance


def eval_for_dev(input_csv, db_path, save_sampled=None, save_performance=None,
                 spd_path=None, random_sample_size=10000, random_state=666, **stratify_config):
    data = prepare_input_for_sampling(input_csv=input_csv, db_path=db_path, spd_path=spd_path,
                                      random_sample_size=random_sample_size, random_state=random_state)

    sampled = stratified_sampling_by_column_value(data, **stratify_config)

    if save_sampled is not None:
        sampled.to_csv(save_sampled)

    sql_db = SqlDB(db_path)
    sampled_with_review = add_review(sampled, sql_db)

    sampled_with_review = add_verdict(sampled_with_review)

    performance = summarise_performance_in_stratified_sampled(sampled_with_review)

    if save_performance is not None:
        performance.to_csv(save_performance)

    return performance
