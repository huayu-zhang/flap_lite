import os
import sys


def get_input_name(msg, msg_2):

    name = input(msg)
    if name.isidentifier():
        return name

    while 1:
        name = input(msg_2)
        if name.isidentifier():
            return name


func_mappings = {
    '1': {
        'name': 'Match addresses to UPRN ID',
        'cmd': "echo 'run matching pipeline'"
    },
    # '2': {
    #     'name': 'Join datasets with addresses',
    #     'cmd': "echo 'run joining pipeline'"
    # },
    # '3': {
    #     'name': 'Parsing addresses',
    #     'cmd': "echo 'run address parsing'"
    #     },
    '4': {
        'name': 'To create and build a database',
        'cmd': "python3 src/database/build_interactive.py %s"
    },
    'exit': {
        'name': 'Exit (or enter "exit")',
        'cmd': "exit"
    }
}


if __name__ == '__main__':

    while 1:
        print('Welcome to FLAP')
        print('Start working in directory %s' % os.getcwd())

        project_name = get_input_name('Please ENTER project name: \n', 'Please ENTER a legal filesystem name: \n')

        project_path = os.path.join(os.getcwd(), 'projects', project_name)

        if os.path.exists(project_path):
            print('Working with Existing project: %s' % project_name)
        else:
            print('Project Setup in: %s' % project_path)
            print('Run check name')
            print('Run project setup pipeline')

        print("Please choose the function you like to use:")

        for k, v in func_mappings.items():
            print("Enter '%s': %s" % (k, v['name']))

        while 1:
            cmd = input('Enter command: \n')

            if cmd == 'exit':
                print('Thanks for using FLAP')
                sys.exit(0)

            elif cmd in func_mappings:
                os.system(func_mappings[cmd]['cmd'] % project_name)

            else:
                print('Command not understood')
