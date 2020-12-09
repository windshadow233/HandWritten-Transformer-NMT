import unicodedata


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    return s


replace = str.maketrans('。？！，；：‘’“”（）【】', '.?!,;:\'\'""()[]')


def full_width2half_width(s):
    return s.translate(replace)
