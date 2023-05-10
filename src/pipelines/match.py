import os
import argparse
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([os.getcwd()])

from src.parser.rule_parser_class import RuleParser
from src.database.sql import SqlDB
from src.project.project import ProjectManager


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', help='Name of the project')
    parser.add_argument('--database_name', help='Name of the database used to match')
    parser.add_argument('--input_file', default=None,
                        help="""Specify the input csv file relative to the project path,
                             if not provided will use the csv file in ./[project_path]/input by default""")
    parser.add_argument('--address_col', default='input_address',
                        help="""The column name for address input. Default 'input_address' """)
    parser.add_argument('--id_col', default='input_id',
                        help="""The column name for the input id. Default 'input_id' """)

    return parser





if __name__ == '__main__':
    args = arg_parser().parse_args()

    input_path = os.path.join(project_config['INPUT_PATH'],
                              os.listdir(project_config['INPUT_PATH'])[0])

    output_path = timestamp_filename(os.path.join(project_config['OUTPUT_PATH'], 'output'))
    # output_path = '/home/huayu_ssh/PycharmProjects/dres_r/projects/dev_1103/output/output-Thu__Mar__16__14_14_32__2023'

    # annotated = pd.read_csv('projects/dev_1103/annotation/annotation-Thu__Jan__19__11_08_37__2023.csv', index_col=0)
    # subset = annotated['input_id']

    if __name__ == '__main__':
        run(input_path, output_path)
