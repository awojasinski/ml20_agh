from typing import Dict


def get_vocabulary_dict() -> Dict[int, str]:
    """Read the fixed vocabulary list from the datafile and return.

    :return: a dictionary of words mapped to their indexes
    """

    path = 'data/vocab.txt'
    dictionary: Dict[int, str] = dict()

    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip().split('\t')
            dictionary[int(word[0])] = word[1]

    return dictionary
