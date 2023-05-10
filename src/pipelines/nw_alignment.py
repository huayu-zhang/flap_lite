import re

from src.utils import tokenize_default, repl_positions
from src.alignment import NeedlemanWunschAlignmentToken, NeedlemanWunschAlignment, SequentialAlignmentResults
from src.distance import damerau_levenshtein_distance


def align_and_score_by_token(s1, s2):
    t1 = tokenize_default(s1)
    t2 = tokenize_default(s2)

    ar = NeedlemanWunschAlignmentToken(seq1=t1, seq2=t2,
                                       match=1, mismatch=-1, gap=0,
                                       allow_fuzzy=True, dist_funct=damerau_levenshtein_distance, dist_tol=0.2,
                                       insertion_label='_').get_result().min_insertion_alignment()
    ar.calculate_score()

    return ar


def align_and_score_by_string(s1, s2):

    ars = NeedlemanWunschAlignment(seq1=s1, seq2=s2).get_result()

    ar = ars.max_score_alignment()

    ar.calculate_score()

    ar.anneal_to_splitter()

    return ar


def align_and_score(s1, s2):

    if s1.isnumeric():
        ar_token = align_and_score_by_token(s1, s2)
        return ar_token

    ar_token = align_and_score_by_token(s1, s2)
    ar_string = align_and_score_by_string(s1, s2)

    return max([ar_token, ar_string], key=lambda x: x.meta['score'])


def align_linear_fields(d, seq):

    ars = []

    for k, v in d.items():
        ar = align_and_score(v, seq)
        ar.meta['name'] = k
        ars.append(ar)

        if k == 'POSTCODE':
            match = re.search(r'([A-Z]{1,2}[0-9][A-Z0-9]?)(?: +)?([0-9][A-Z]{2})', seq)
            if match is not None:
                start, end = match.span()
                pys = list(range(start, end))
                seq = repl_positions(seq, pys)

        elif ar.meta['score'] > 0:

            if ar.meta['perc_x'] > 0.3:
                seq = repl_positions(seq, [py for _, py in ar.meta['mapping']])

    return SequentialAlignmentResults(ars)


# d = {
#     'BUILDING_NUMBER': '9',
#     'THOROUGHFARE': 'BRUNSWICK RD',
#     'POST_TOWN': 'EDINBURGH',
#     'POSTCODE': 'EH7 5GX'
# }
#
# seq = '9 BRUNSWICK ROAD EDINBURGH EH7 5GX'
#
#
# ars = align_linear_fields(d, seq)
#
# ars.get_alignment_features()


# l1 = ' '.join(['ST', 'LEONARD', 'ST', 'PATRICK', 'ST', 'ANDREWS'])
# l2 = ' '.join(['ST', 'ANDREWS'])
# #
# #
# print(align_and_score_by_token(l1, l2))
#
# print(align_and_score_by_string(l1, l2))
#
# print(align_and_score(l1, l2))
#
#
#
# nw = NeedlemanWunschAlignment('FLAT 4', 'F4 ** ************* ********* LOTHIAN *******\n')
# ars = nw.get_result()
#
# print(ars.max_score_alignment())
#
#
# for ar in ars.alignment_results:
#     print(ar)
#     ar.calculate_score()
#     print(ar.meta)

# #
# print(align_and_score('FLAT 4', 'F4 ** ************* ********* LOTHIAN *******\n'))

