from .rule_parser import rule_parsing


def span_to_string(span, s):
    start, end = span
    return s[start:end]


def parsing_to_string(parsing, s):
    return {k: ' '.join([span_to_string(sub_s, s) for sub_s in v])
            for k, v in parsing.items() if len(v)}


class Parser:

    def __init__(self, address):
        self.address = address
        self.parsed = None

    def parse_by_rules(self):
        self.parsed = rule_parsing(self.address)
        return self.parsed

