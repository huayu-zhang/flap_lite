def span_to_string(span, s):
    start, end = span
    return s[start:end]


def parsing_to_string(parsing, s):
    return {k: ' '.join([span_to_string(sub_s, s) for sub_s in v])
            for k, v in parsing.items() if len(v)}

