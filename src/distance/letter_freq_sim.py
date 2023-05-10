import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter


def dict_update_existingkeys(dict_to_update, new_dict):
    dict_to_update.update((k, new_dict[k]) for k in dict_to_update.keys())
    return dict_to_update


def letter_freq(s):
    LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    LETTERS_DICT = {key: 0 for key in LETTERS}
    freq = dict_update_existingkeys(LETTERS_DICT, Counter(s))
    return freq


def letter_freq_cosine_sim(s1, s2):
    letter_dict1 = letter_freq(s1)
    letter_dict2 = letter_freq(s2)

    uu = np.array(list(letter_dict1.values()))
    vv = np.array(list(letter_dict2.values()))

    if uu @ vv:
        sim = 1 - cosine(uu, vv)
        return sim if not np.isnan(sim) else 0

    else:
        return 0
