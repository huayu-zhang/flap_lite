import os
import json
import uuid
import time
import pandas as pd

from src.pipelines.tree_align import RuleMatchByTree
from src.pipelines.output_summary import OutputSummary


def _mkdir_if_not_exist(path):

    if not os.path.exists(path):
        os.mkdir(path)


class ProjectManager:

    def __init__(self):
        self.project_path = os.path.join(os.getcwd(), 'projects')
        self.project_names = [file for file in os.listdir(self.project_path)
                              if os.path.isdir(os.path.join(self.project_path, file))]

    def update_list_of_project_names(self):
        self.project_names = [file for file in os.listdir(self.project_path)
                              if os.path.isdir(os.path.join(self.project_path, file))]

    def get_project_names(self):
        return self.project_names

    def get_project_path(self, project_name):
        return os.path.join(self.project_path, project_name)

    def create_project(self, project_name):
        if project_name in self.project_names:
            print('Project %s exists, no new project created.' % project_name)
        else:
            new_project_dir = os.path.join(self.project_path, project_name)
            _mkdir_if_not_exist(new_project_dir)

            init_config = {
                'paths': {
                    'input': os.path.join(new_project_dir, 'input'),
                    'output_raw': os.path.join(new_project_dir, 'output_raw'),
                    'output_summary': os.path.join(new_project_dir, 'output_summary'),
                    'output_formatted': os.path.join(new_project_dir, 'output_formatted'),
                    'annotation': os.path.join(new_project_dir, 'annotation'),
                    'config': os.path.join(new_project_dir, 'config'),
                    'log': os.path.join(new_project_dir, 'log')
                },
                'files': {
                    'config': os.path.join(new_project_dir, 'config', 'config.json'),
                    'log': os.path.join(new_project_dir, 'log', 'log.json')
                }
            }

            for _, path in init_config['paths'].items():
                _mkdir_if_not_exist(path)

            with open(init_config['files']['config'], 'w') as f:
                json.dump(init_config, f, indent=4)

            init_log = [
                {
                    'event_id': str(uuid.uuid4()),
                    'event_name': 'project_creation',
                    'params': {
                        'project_name': project_name
                    },
                    'time': time.asctime()
                }
            ]

            with open(init_config['files']['log'], 'w') as f:
                json.dump(init_log, f, indent=4)

            print('Project created at: %s' % new_project_dir)

        self.update_list_of_project_names()


class Project:

    def __init__(self, project_name):

        self.project_name = project_name
        self.project_path = os.path.join(os.getcwd(), 'projects', project_name)

        self.sub_paths = {
                    'input': os.path.join(self.project_path, 'input'),
                    'input_file': os.path.join(self.project_path, 'input', 'input_df.csv'),
                    'output_raw': os.path.join(self.project_path, 'output_raw'),
                    'output_summary': os.path.join(self.project_path, 'output_summary'),
                    'output_formatted': os.path.join(self.project_path, 'output_formatted'),
                    'output_formatted_file': os.path.join(self.project_path, 'output_formatted', 'results.csv'),
                    'annotation': os.path.join(self.project_path, 'annotation'),
                    'config_file': os.path.join(self.project_path, 'config', 'config.json'),
                    'log_file': os.path.join(self.project_path, 'log', 'log.json')
                }

    def match(self, sql_db):

        matcher = RuleMatchByTree(sql_db)

        run(self.sub_paths['input_file'], self.sub_paths['output_raw'], matcher=matcher)

        output_summary = OutputSummary(
            self.sub_paths['output_raw'])

        output_summary.save_summary(self.sub_paths['output_summary'])
        output_summary.save_result(self.sub_paths['output_formatted_file'])


def _run_task(args):
    address, input_id, output_path, matcher = args

    output_file = os.path.join(output_path, '%s.json' % input_id)

    if not os.path.exists(output_file):
        matcher.match(address=address, input_id=input_id)

        log = matcher.log

        with open(output_file, 'w') as f:
            json.dump(log, f, indent=4)


def run(input_path, output_path, matcher, subset=None, parallel=True):

    input_df = pd.read_csv(input_path)

    if subset is not None:
        input_df = input_df.loc[input_df['input_id'].isin(subset)]

    task_generator = ((row['input_address'], row['input_id'], output_path, matcher) for _, row in input_df.iterrows())

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if parallel:
        from tqdm.contrib.concurrent import process_map
        process_map(_run_task, task_generator)
    else:
        try:
            from tqdm import tqdm
            for args in tqdm(task_generator):
                _run_task(args)
        except ModuleNotFoundError:
            for args in task_generator:
                _run_task(args)
