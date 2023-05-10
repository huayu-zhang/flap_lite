import os
import json
import pandas as pd
import textwrap

from src.utils import join_uprn_fields, timestamp_filename


def read_output_file(file):
    with open(file, 'r') as f:
        log = json.load(f)
    return log


def lite_summary(log):
    summary = {
        'task_id': log['logging']['task_id'],
        'input_address': '\n'.join(textwrap.wrap(str(log['logging']['input_address']), width=20)),
        'uprn_fields': join_uprn_fields(log['db_record']) if 'db_record' in log else '',
        'input_id': log['logging']['input_id'],
        'db_id': log['db_record']['ex_uprn'] if 'db_record' in log else '',
        'UPRN': log['db_record']['UPRN'] if 'db_record' in log else '',
        'error': log['error'] if 'error' in log else ''
    }
    return summary


class OutputSummary:

    def __init__(self, output_path):
        self.output_path = output_path
        self.files = [os.path.join(self.output_path, file)
                      for file in os.listdir(output_path)]
        self.logs = None
        self.summary = None

    def load_logs(self):

        if self.logs is None:
            try:
                from tqdm import tqdm
                self.logs = [read_output_file(file) for file in tqdm(self.files)]
            except ModuleNotFoundError:
                self.logs = [read_output_file(file) for file in self.files]

    def get_lite_summary(self):

        if self.logs is None:
            self.load_logs()

        if self.summary is not None:
            return self.summary

        try:
            from tqdm import tqdm
            self.summary = pd.DataFrame.from_records([lite_summary(log) for log in tqdm(self.logs)])
        except ModuleNotFoundError:
            self.summary = pd.DataFrame.from_records([lite_summary(log) for log in self.logs])

        self.summary['correct'] = pd.NA
        self.summary['manual_db_id'] = pd.NA
        self.summary['not_in_db'] = pd.NA
        self.summary['too_broad'] = pd.NA
        self.summary['low_quality'] = pd.NA
        self.summary['remark'] = pd.NA

        return self.summary

    def save_summary(self, path, chunk_size=5000):

        if self.summary is None:
            self.get_lite_summary()

        # path = os.path.join(path, 'output_summary-%s' % os.path.basename(self.output_path))

        if not os.path.exists(path):
            os.mkdir(path)

        for start in range(0, self.summary.shape[0], chunk_size):
            self.summary.iloc[start:start + chunk_size].to_csv(
                os.path.join(path, 'output_summary_%s-%s.csv') % (start, start + chunk_size))

    def save_result(self, path):

        if self.summary is None:
            self.get_lite_summary()

        res = self.summary[['input_id', 'UPRN']]

        res.to_csv(path)
